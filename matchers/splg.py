import argparse
import os

import h5py
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('--recalc_features', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def extract_features(args):
    name_path = os.path.join(args.out_path, args.name)

    cache_path = f'{name_path}-extracted_sp_{args.max_features}_{args.resize if args.resize is not None else "noresize"}.pt'

    if os.path.exists(cache_path) and not args.recalc_features:
        print(f"Features already found in {cache_path}")
        return torch.load(cache_path)

    image_list_path = f'{name_path}_image_list.txt'
    with open(image_list_path, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]

    extractor = SuperPoint(max_num_keypoints=args.max_features).eval().cuda()

    print("Extracting features")
    feature_dict = {}

    for img_name in tqdm(image_list):
        img_path = os.path.join(args.dataset_path, img_name)
        image_tensor = load_image(img_path).cuda()
        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[img_name] = kp_tensor

    torch.save(feature_dict, cache_path)
    print("Features saved to: ", cache_path)

    return feature_dict


def match_pairs(feature_dict, args):
    name_path = os.path.join(args.out_path, args.name)

    image_list_path = f'{name_path}_image_pairs.txt'
    with open(image_list_path, 'r') as f:
        pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

    matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()

    f = h5py.File(f'{name_path}_splg_{args.max_features}_{args.resize if args.resize is not None else "noresize"}.h5', 'w')

    print("Matching features")
    with torch.no_grad():
        for img_name_1, img_name_2 in tqdm(pair_list):
            feats_1 = feature_dict[img_name_1]
            feats_2 = feature_dict[img_name_2]
            out = matcher({'image0': feats_1, 'image1': feats_2})
            feats_1, feats_2, out = [rbd(x) for x in [feats_1, feats_2, out]]  # remove batch dimension
            matches = out['matches']
            score = out['scores']
            points_1 = feats_1['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points_2 = feats_2['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

            kpts_1 = points_1.cpu().numpy()
            kpts_2 = points_2.cpu().numpy()
            score = score.cpu().numpy()

            match_positions = np.hstack([kpts_1[:, :2], kpts_2[:, :2], score[:, np.newaxis]])
            f.create_dataset(f"{img_name_1}-{img_name_2}", data=match_positions, compression='gzip', chunks=True)
    f.close()


def match_splg(args):
    feature_dict = extract_features(args)
    match_pairs(feature_dict, args)


if __name__ == '__main__':
    args = parse_args()
    match_splg(args)
