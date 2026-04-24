import argparse
import copy
import os
from argparse import Namespace
from pathlib import PureWindowsPath, Path

import cv2
import h5py
import numpy as np
import matplotlib

from utils.results import get_mde_list, compute_auc
from utils.storage import load_full_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st', '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-sw', '--sampson_weight', type=float, default=1.0)
    parser.add_argument('-rt', '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('--matches', type=str, default='splg_2048_noresize')
    parser.add_argument('-n', '--n_pairs', type=int, default=20)
    parser.add_argument('--work_path')
    parser.add_argument('--data_path')
    parser.add_argument('--name')

    return parser.parse_args()


def get_worst_pairs(args):
    work_path = args.work_path
    name = args.name

    gt_args = copy.copy(args)
    gt_args.depth = 'gt'
    gt_results = load_full_results(gt_args)

    baseline_by_pair = {}
    for r in gt_results:
        if r['experiment'] == 'baseline_calib' and r['iterations'] == 1000:
            pair = (r['image_name_1'], r['image_name_2'])
            err = max(r['R_err'], r['t_err'])
            baseline_by_pair[pair] = 180.0 if np.isnan(err) else err

    pairs = list(baseline_by_pair.keys())

    pair_ids = {pair: index for index, pair in enumerate(pairs)}

    mde_list = get_mde_list(name, work_path)[:1]

    p_errs = np.full([len(pairs), len(mde_list) + 1], np.nan)
    inliers = np.empty([len(pairs), len(mde_list) + 1], dtype=object)
    shift_solver = np.empty([len(pairs), len(mde_list) + 1], dtype=bool)

    for i, pair in enumerate(pairs):
        p_errs[i, -1] = baseline_by_pair[pair]

    for mde_id, mde in enumerate(mde_list):
        mde_args = copy.copy(args)
        mde_args.depth = mde
        try:
            mde_results = load_full_results(mde_args)
        except Exception as e:
            print(f"Skipping {mde}: {e}")
            continue

        iters_results = [r for r in mde_results
                         if r['iterations'] == 1000
                         and r['experiment'] in ['calib', 'calib_shift']]

        for r in iters_results:
            p_err = max(r['R_err'], r['t_err'])
            pair = (r['image_name_1'], r['image_name_2'])
            pair_id = pair_ids[pair]
            inliers = r['info']['inliers']
            if np.isnan(p_errs[pair_id, mde_id]):
                p_errs[pair_id, mde_id] = p_err
                inliers[pair_id, mde_id] = inliers
                shift_solver[pair_id, mde_id] = r['experiment'] == 'calib_shift'
            elif p_errs[pair_id, mde_id] > p_err:
                p_errs[pair_id, mde_id] = p_err
                shift_solver[pair_id, mde_id] = r['experiment'] == 'calib_shift'
                inliers[pair_id, mde_id] = inliers

    difference = p_errs[:, :-1] - p_errs[:, -1:]
    best = np.nanmin(difference, axis=1)
    worst_best = np.argpartition(-best, 20)[:20]

    worst_pairs_dict = {}

    for id in worst_best:
        pair = pairs[id]
        worst_pairs_dict[pair] = {'results': {}}
        worst_pairs_dict[pair]['results']['gt'] = {'p_err': p_errs[id, -1], 'solver': 'baseline'}
        for mde_id, mde in enumerate(mde_list):
             worst_pairs_dict[pair]['results'][mde] = {'p_err': p_errs[id, mde_id],
                                                       'solver': 'calib_shift' if shift_solver[id, mde_id] else 'calib'}

    return worst_pairs_dict


def get_matches(worst_pairs_dict, args):
    matches_path = os.path.join(args.work_path, f'{args.name}__{args.matches}')
    with h5py.File(matches_path) as f:
        for pair, d in worst_pairs_dict.items():
            kp = np.array(f[f'{pair[0]}-{pair[1]}'])
            # do not forget to convert to list later!
            kp1 = kp[:, :2]
            kp2 = kp[:, :2]
            d['kp1'] = kp1
            d['kp2'] = kp2


# taken from https://github.com/microsoft/MoGe/blob/main/moge/utils/vis.py
def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def enforce_max_width(image, max_width=512):
    h,w = image.shape[:2]
    scale = max_width / max(h, w)
    new_h = int(h * scale)
    return cv2.resize(image, (max_width, new_h))


def export_images(worst_pairs_dict, args):
    name = args.name
    work_path = args.work_path
    dataset_path = args.data_path

    images = [x[0] for x in worst_pairs_dict.keys()]
    images.extend([x[1] for x in worst_pairs_dict.keys()])
    images = list(set(images))
    image_ids = {image: i for i, image in enumerate(images)}

    mde_list = worst_pairs_dict[list(worst_pairs_dict.keys())[0]]['results'].keys()
    mde_ids = {mde: i for i, mde in enumerate(mde_list)}

    for pair, d in worst_pairs_dict.items():
        d['exported_rgb_image1_path'] = f'image_{image_ids[pair[0]]}.png'
        d['exported_rgb_image2_path'] = f'image_{image_ids[pair[1]]}.png'

        for mde, mde_dict in d['results'].items():
            if mde == 'gt':
                continue
            mde_dict['exported_rgb_image1_path'] = f'depth_{image_ids[pair[0]]}_{mde_ids[mde]}.png'
            mde_dict['exported_rgb_image2_path'] = f'depth_{image_ids[pair[1]]}_{mde_ids[mde]}.png'


    worst_examples_dir = os.path.join(work_path, 'examples')
    os.makedirs(worst_examples_dir, exist_ok=True)

    for mde_id, mde in enumerate(mde_list):
        if mde == 'gt':
            continue
        with h5py.File(os.path.join(work_path, f'{name}_depth_{mde}.h5'), 'r') as f:
            for i, image_name in enumerate(images):
                depth = np.array(f[f'{image_name}_depth'])
                depth_color = colorize_depth(depth)

                save_image_path = os.path.join(worst_examples_dir, f'depth_{i}_{mde_id}.png')
                cv2.imwrite(save_image_path, enforce_max_width(depth_color))

    for i, image_name in enumerate(images):
        image = cv2.imread(os.path.join(dataset_path, Path(PureWindowsPath(image_name))))
        save_image_path = os.path.join(worst_examples_dir, f'image_{i}.png')
        cv2.imwrite(save_image_path, enforce_max_width(image))

    return worst_pairs_dict



if __name__ == '__main__':
    args = parse_args()
    worst_pairs_dict = get_worst_pairs(args)
    worst_pairs_dict = get_matches(worst_pairs_dict, args)
    worst_pairs_dict = export_images(worst_pairs_dict, arg)
