import argparse
import collections
import itertools
import ntpath
import os
from pathlib import Path
import random

import h5py
import numpy as np

from scipy.spatial.transform import Rotation
from tqdm import tqdm

from datasets.colmap_utils import cam_to_K, read_model
from utils.system_info import save_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=100)
    parser.add_argument('-mi', '--max_images', type=int, default=None, help='Maximum images per scene')
    parser.add_argument('-mp', '--max_pairs', type=int, default=None, help='Maximum pairs per scene')
    parser.add_argument('-mr', '--max_resolution', type=int, default=None, help='Maximum resolution in the larger side of the image')
    parser.add_argument('--min_keypoint_overlap', type=int, default=20, help='Minimum number of gt 3D keypoints overlap')
    parser.add_argument('--min_area_overlap', type=float, default=0.1, help='Minimum overlap area based on gt 3D keypoints')
    parser.add_argument('--name', type=str, default='dataset', help='Path to dataset files')
    parser.add_argument('--check_images', action='store_true', default=False, help='Keep only images that are actually available on disk')
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def get_area(pts):
    width = np.max(pts[:, 0]) - np.min(pts[:, 0])
    height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    return width * height

def get_overlap_areas(image_1, image_2, cameras):
    cam_1 = cameras[image_1.camera_id]
    cam_2 = cameras[image_2.camera_id]

    overlap = set(image_1.pts_dict.keys()).intersection(set(image_2.pts_dict.keys()))
    overlap = list(overlap)
    if len(overlap) < 2:
        return 0, 0.0

    pts_1 = np.array([image_1.pts_dict[x] for x in overlap])
    pts_2 = np.array([image_2.pts_dict[x] for x in overlap])
    area_1 = get_area(pts_1) / (cam_1.width * cam_1.height)
    area_2 = get_area(pts_2) / (cam_2.width * cam_2.height)

    return len(overlap), min(area_1, area_2)


def get_dataset_paths(basename, dataset_path, subset):
    subset_path = os.path.join(dataset_path, subset)
    if basename.lower() == 'phototourism' or 'pt' == basename.lower():
        model_path = os.path.join('dense', 'sparse')
        img_path = os.path.join('dense', 'images')
    elif basename.lower() == 'eth3d_depth' or basename.lower() == 'scannetpp':
        model_path = os.path.join('colmap_gt')
        img_path = os.path.join('images')
    elif basename.lower() == 'urban':
        model_path = os.path.join('sfm')
        img_path = os.path.join('images_full_set')
    elif 'aachen' in basename.lower():
        model_path = os.path.join('3D-models/aachen_v_1_1')
        img_path = os.path.join('images_upright')
    elif 'multiview_undistorted' in basename.lower() or 'eth3d' in basename.lower():
        model_path = os.path.join('dslr_calibration_undistorted')
        img_path = os.path.join('images')
    else:
        model_path = os.path.join('sfm')
        img_path = os.path.join('images_full')
    return img_path, model_path, subset_path


def enforce_max_resolution(camera, args):
    w = camera.width
    h = camera.height
    K = cam_to_K(camera)

    if args.max_resolution is None or max(w,h) < args.max_resolution:
        return w, h, K, w, h, K

    if w > h:
        scale = args.max_resolution / w
    else:
        scale = args.max_resolution / h

    new_w, new_h = int(scale * w), int(scale * h)
    new_K = np.copy(K)
    new_K[0, :] *= new_w / w
    new_K[1, :] *= new_h / h

    return new_w, new_h, new_K, w, h, K



def create_gt_h5(model, image_ids, subset, f, f_txt, args):
    images = model['images']
    cameras = model['cameras']
    full_img_path = ntpath.normpath(os.path.join(subset, model['img_path']))

    for img_id in image_ids:
        img = images[img_id]
        camera = cameras[img.camera_id]
        name = ntpath.normpath(ntpath.join(full_img_path, img.name))

        f_txt.write(name + '\n')

        q = img.qvec
        t = img.tvec
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        w, h, K, w_orig, h_orig, K_orig = enforce_max_resolution(camera, args)
        size = np.array([int(w), int(h)], dtype=int)
        size_orig = np.array([int(w_orig), int(h_orig)], dtype=int)

        K_name = f'{name}_K'
        R_name = f'{name}_R'
        T_name = f'{name}_T'
        size_name = f'{name}_size'

        f.create_dataset(R_name, data=R, compression='gzip', chunks=True)
        f.create_dataset(T_name, data=t, compression='gzip', chunks=True)
        f.create_dataset(size_name, data=size, compression='gzip', chunks=True)
        f.create_dataset(K_name, data=K, compression='gzip', chunks=True)
        f.create_dataset(size_name + '_orig', data=size_orig, compression='gzip', chunks=True)
        f.create_dataset(K_name + '_orig', data=K_orig, compression='gzip', chunks=True)


def valid_pairs(model, image_ids, args):
    pairs = []

    cameras = model['cameras']

    for img_id_1, img_id_2 in tqdm(itertools.combinations(image_ids, 2),
                                   total=len(image_ids) * (len(image_ids) - 1) // 2):
        keypoints_overlap, area_overlap = get_overlap_areas(model['images'][img_id_1], model['images'][img_id_2], cameras)

        if keypoints_overlap < args.min_keypoint_overlap or area_overlap < args.min_area_overlap:
            continue

        pairs.append((img_id_1, img_id_2))

        # name_1 = ntpath.normpath(ntpath.join(full_img_path, img_1.name))
        # name_2 = ntpath.normpath(ntpath.join(full_img_path, img_2.name))
        # overlap_f.write(f"{name_1},{name_2},{keypoints_overlap},{area_overlap:.4f}\n")
    return pairs



def enforce_max_images_pairs(model, args):
    if args.max_images is None or len(model['images']) <= args.max_images:
        # if there are fewer than max images we keep all
        image_ids = list(model['images'].keys())
        pairs = valid_pairs(model, image_ids, args)
    elif args.max_pairs is None:
        # if we don't care about max pairs just choose randomly
        image_ids = random.sample(list(model['images'].keys()), args.max_images)
        pairs = valid_pairs(model, image_ids, args)
    else:
        # if there are more we ensure that we select such images that we can get 2 * max_pairs to choose from
        image_ids = random.sample(list(model['images'].keys()), args.max_images)
        pairs = valid_pairs(model, image_ids, args)
        while len(pairs) < 2 * args.max_pairs:
            print("Not enough pairs. Retrying with random image subset!")
            image_ids = random.sample(model['images'].keys(), args.max_images)
            pairs = valid_pairs(model, image_ids, args)

    return image_ids, pairs


def order_pairs_approx(pairs):
    id_freq = collections.Counter(id_ for pair in pairs for id_ in pair)

    # Weight inversely proportional to frequency
    weights =np.array([1.0 / (id_freq[p[0]] + id_freq[p[1]]) for p in pairs])
    weights /= np.sum(weights)

    # Weighted sample without replacement for the first max_pairs
    indices = np.random.choice(range(len(pairs)), p=weights, size=args.max_pairs, replace=False)

    return [pairs[i] for i in indices]


def create_gt_pairs(model, pairs, subset, f_pairs):
    images = model['images']
    full_img_path = ntpath.normpath(os.path.join(subset, model['img_path']))

    for img_id_1, img_id_2 in pairs:
        img_1 = images[img_id_1]
        name_1 = ntpath.normpath(ntpath.join(full_img_path, img_1.name))
        img_2 = images[img_id_2]
        name_2 = ntpath.normpath(ntpath.join(full_img_path, img_2.name))
        f_pairs.write(f'{name_1},{name_2}\n')


def process_subsets(args, subsets):
    out_path = os.path.join(args.out_path, args.name)
    h5_path = f'{out_path}.h5'
    f = h5py.File(h5_path, 'w')
    print(f"Writing GT info to {h5_path}")
    save_metadata(f)

    txt_path = f'{out_path}_image_list.txt'
    f_txt = open(txt_path, 'w')
    print(f"Writing list of images info to {txt_path}")

    # splg_h5_path = f'{out_path}_pairs_splg.h5'
    # f_splg = open(splg_h5_path, 'w')
    #
    # roma_h5_path = f'{out_path}_pairs_roma.h5'
    # f_roma= open(roma_h5_path, 'w')

    pairs_path = f'{out_path}_image_pairs.txt'
    f_pairs = open(pairs_path, 'w')

    for subset in subsets:
        print(f"Processing subset: {subset}")
        model = get_model(args, subset)

        image_ids, pairs = enforce_max_images_pairs(model, args)

        create_gt_h5(model, image_ids, subset, f, f_txt, args)

        if args.max_pairs is not None and len(pairs) > args.max_pairs:
            pairs = order_pairs_approx(pairs)[:args.max_pairs]

        create_gt_pairs(model, pairs, subset, f_pairs)
        del model

    f.close()
    f_txt.close()
    f_pairs.close()
    # f_roma.close()
    # f_splg.close()



def get_model(args, subset):
    model = {}
    dataset_path = Path(args.dataset_path)
    basename = os.path.basename(dataset_path)
    print(f"Basename: {basename}")
    img_path, model_path, subset_path = get_dataset_paths(basename, dataset_path, subset)
    print(f"Reading model at {os.path.join(subset_path, model_path)}")
    cameras, images, points = read_model(os.path.join(subset_path, model_path))

    if args.check_images:
        prev_images_len = len(images)
        full_img_path = os.path.join(args.dataset_path, subset, img_path)
        images = {k: v for k, v in images.items()
                  if os.path.exists(os.path.normpath(os.path.join(full_img_path, v.name)))}

        print(f"Model has {prev_images_len} images, but only {len(images)} found in {full_img_path}. "
              f"Using only found images.")

    # add dict for easier computation of overlap
    print(f"Processing 3D points")
    for image in tqdm(images.values()):
        pts_dict = {}
        for id, xy in zip(image.point3D_ids, image.xys):
            if id == -1:
                continue
            pts_dict[id] = xy
        image.pts_dict = pts_dict


    model['cameras'] = cameras
    model['images'] = images
    model['points'] = points
    model['img_path'] = img_path
    model['model_path'] = model_path
    model['points'] = points
    return model


def process_colmap_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.name is None:
        args.name = os.path.basename(args.dataset_path)


    out_dir = os.path.join(args.out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset_path = Path(args.dataset_path)
    dir_list = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]

    print("Found subsets: ", dir_list)
    process_subsets(args, dir_list)


if __name__ == '__main__':
    args = parse_args()
    process_colmap_dataset(args)