import argparse
import copy
import json
import os
import shutil
import tarfile
from argparse import Namespace
from pathlib import PureWindowsPath, Path

import cv2
import h5py
import numpy as np
import matplotlib
import submitit
from tqdm import tqdm

from utils.config import config_iterator
from utils.geometry import get_kp_depth
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
    parser.add_argument('--work_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, default='')
    parser.add_argument('--config_path', type=str, default=None)

    parser.add_argument('--account', type=str, default='p1358-25-2',
                        help='Slurm account name')
    parser.add_argument('--queue', type=str, default='short',
                        help='Slurm partition/queue name')
    parser.add_argument('--mem_gb', type=int, default=32,
                        help='Memory per job in GB')
    parser.add_argument('--timeout_min', type=int, default=60,
                        help='Expected max runtime per job in minutes')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for submitit logs (default: <work_path>/slurm_logs)')
    parser.add_argument('-nw', '--num_workers', type=int, default=8)

    return parser.parse_args()


def get_worst_pairs(args):
    work_path = args.work_path
    name = args.name

    gt_args = copy.copy(args)
    gt_args.depth = 'none'
    gt_results = load_full_results(gt_args)

    baseline_by_pair = {}
    inliers_by_pair = {}
    for r in gt_results:
        if r['experiment'] == 'baseline_calib' and r['iterations'] == 1000:
            if len(r['info']['inliers']) < 6:
                continue
            pair = (r['image_name_1'], r['image_name_2'])
            err = max(r['R_err'], r['t_err'])
            baseline_by_pair[pair] = 180.0 if np.isnan(err) else err
            inliers_by_pair[pair] = r['info']['inliers']

    pairs = list(baseline_by_pair.keys())
    pair_ids = {pair: index for index, pair in enumerate(pairs)}
    mde_list = get_mde_list(name, work_path)

    p_errs = np.full([len(pairs), len(mde_list) + 1], np.nan)
    inliers = np.empty([len(pairs), len(mde_list) + 1], dtype=object)
    # shift_solver = np.empty([len(pairs), len(mde_list) + 1], dtype=bool)

    for i, pair in enumerate(pairs):
        p_errs[i, -1] = baseline_by_pair[pair]
        inliers[i, -1] = inliers_by_pair[pair]
        # shift_solver[i, -1] = False

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
                         # and r['experiment'] in ['calib', 'calib_shift']]
                         and r['experiment'] == 'calib']

        for r in iters_results:
            pair = (r['image_name_1'], r['image_name_2'])
            if pair not in pair_ids:
                continue
            p_err = max(r['R_err'], r['t_err'])
            pair_id = pair_ids[pair]
            r_inliers = r['info']['inliers']
            if np.isnan(p_errs[pair_id, mde_id]):
                p_errs[pair_id, mde_id] = p_err
                inliers[pair_id, mde_id] = r_inliers
                # shift_solver[pair_id, mde_id] = r['experiment'] == 'calib_shift'

                # print('******')
                # print(len(r_inliers), len(inliers[pair_id, mde_id]),len(inliers[pair_id, -1]))
                # print(np.sum(r_inliers), np.sum(inliers[pair_id, -1]))

            elif p_errs[pair_id, mde_id] > p_err:
                p_errs[pair_id, mde_id] = p_err
                # shift_solver[pair_id, mde_id] = r['experiment'] == 'calib_shift'
                inliers[pair_id, mde_id] = r_inliers

                # print('******')
                # print(len(r_inliers), len(inliers[pair_id, mde_id]),len(inliers[pair_id, -1]))
                # print(np.sum(r_inliers), np.sum(inliers[pair_id, -1]))


    print("Calculating worst pairs")
    gt_mask = p_errs[:, -1] < 10.0
    difference = p_errs[:, :-1] - p_errs[:, -1:]
    best = np.where(gt_mask, np.nanmin(difference, axis=1), np.nan)
    valid_indices = np.where(~np.isnan(best))[0]
    worst_best = valid_indices[np.argpartition(-best[valid_indices], min(args.n_pairs, len(valid_indices)))[:min(args.n_pairs, len(valid_indices))]]

    worst_pairs_dict = {}

    for id in worst_best:
        pair = pairs[id]
        best_mde_id = int(np.nanargmin(p_errs[id, :-1]))
        worst_pairs_dict[pair] = {
            'baseline_p_err': p_errs[id, -1],
            'best_mde_p_err': p_errs[id, best_mde_id],
            'best_mde': mde_list[best_mde_id],
            'results': {},
        }
        worst_pairs_dict[pair]['results']['gt'] = {'p_err': p_errs[id, -1], 'solver': 'baseline', 'inliers': inliers[id, -1]}
        for mde_id, mde in enumerate(mde_list):
             worst_pairs_dict[pair]['results'][mde] = {'p_err': p_errs[id, mde_id],
                                                       # 'solver': 'calib_shift' if shift_solver[id, mde_id] else 'calib',
                                                       'solver': 'calib',
                                                       'inliers': inliers[id, mde_id]}

    return worst_pairs_dict


def get_matches(worst_pairs_dict, args):
    matches_path = os.path.join(args.work_path, f'{args.name}_{args.matches}.h5')
    with h5py.File(matches_path) as f:
        for pair, d in worst_pairs_dict.items():
            kp = np.array(f[f'{pair[0]}-{pair[1]}'])
            # do not forget to convert to list later!
            kp1 = kp[:, :2]
            kp2 = kp[:, 2:]
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


def export_images(worst_pairs_dict, save_dir, args):
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
        d['image1_id'] = image_ids[pair[0]]
        d['image2_id'] = image_ids[pair[1]]
        d['exported_rgb_image1_path'] = f'images/image_{image_ids[pair[0]]}.png'
        d['exported_rgb_image2_path'] = f'images/image_{image_ids[pair[1]]}.png'

        for mde, mde_dict in d['results'].items():
            if mde == 'gt':
                continue
            mde_dict['exported_depth_image1_path'] = f'depths/depth_{image_ids[pair[0]]}_{mde_ids[mde]}.png'
            mde_dict['exported_depth_image2_path'] = f'depths/depth_{image_ids[pair[1]]}_{mde_ids[mde]}.png'

    images_dir = os.path.join(save_dir, 'images')
    depths_dir = os.path.join(save_dir, 'depths')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    print("Saving depths")
    for mde_id, mde in tqdm(enumerate(mde_list), total=len(mde_list)):
        if mde == 'gt':
            continue
        with h5py.File(os.path.join(work_path, f'{name}_depth_{mde}.h5'), 'r') as f:
            for i, image_name in enumerate(images):
                depth = np.array(f[f'{image_name}_depth'])
                depth_color = colorize_depth(depth)
                save_image_path = os.path.join(depths_dir, f'depth_{i}_{mde_id}.png')
                cv2.imwrite(save_image_path, enforce_max_width(depth_color))

                for dd in worst_pairs_dict.values():
                    if dd['image1_id'] == i:
                        dd['results'][mde]['d1'] = get_kp_depth(dd['kp1'], depth, interpolation='nearest')
                    if dd['image2_id'] == i:
                        dd['results'][mde]['d2'] = get_kp_depth(dd['kp2'], depth, interpolation='nearest')

    print("Saving images")
    for i, image_name in tqdm(enumerate(images)):
        image = cv2.imread(os.path.join(dataset_path, Path(PureWindowsPath(image_name))))
        save_image_path = os.path.join(images_dir, f'image_{i}.png')
        resized_image = enforce_max_width(image)
        cv2.imwrite(save_image_path, resized_image)

        for dd in worst_pairs_dict.values():
            if dd['image1_id'] == i:
                dd['kp1'][:, 0] *= resized_image.shape[1] / image.shape[1]
                dd['kp1'][:, 1] *= resized_image.shape[0] / image.shape[0]
            if dd['image2_id'] == i:
                dd['kp2'][:, 0] *= resized_image.shape[1] / image.shape[1]
                dd['kp2'][:, 1] *= resized_image.shape[0] / image.shape[0]


    for pair, d in worst_pairs_dict.items():
        for mde, mde_dict in d['results'].items():
            if mde == 'gt':
                mde_dict['unused_kps'] = []
                continue

            d1 = mde_dict['d1']
            d2 = mde_dict['d2']
            l = np.logical_and(np.isfinite(d1), np.isfinite(d2))
            l = np.logical_and(d1 > 0, l)
            l = np.logical_and(d2 > 0, l)
            mde_dict['unused_kps'] = np.argwhere(~l)

    # print(worst_pairs_dict)


def save_json(worst_pairs_dict, save_dir, args):
    examples_dir = os.path.join(save_dir, 'examples')
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    def convert(obj, key=None):
        if isinstance(obj, dict):
            return {(f'{k[0]}-{k[1]}' if isinstance(k, tuple) else k): convert(v, k) for k, v in obj.items() if k not in ('d1', 'd2')}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            if key in ('kp1', 'kp2'):
                return [[round(float(x), 2) for x in row] for row in obj]
            lst = obj.ravel().tolist()
            if key == 'inliers':
                return [int(x) for x in lst]
            return lst
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj

    main_index = {}
    for pair, d in worst_pairs_dict.items():
        pair_key = f'{pair[0]}-{pair[1]}'.replace('/', '_').replace('\\', '_')
        pair_json_name = f'{pair_key}.json'
        pair_json_path = os.path.join(results_dir, pair_json_name)
        with open(pair_json_path, 'w') as f:
            json.dump(convert(d), f)
        top_level = {k: convert(v, k) for k, v in d.items() if k != 'results'}
        top_level['result_path'] = f'results/{pair_json_name}'
        main_index[f'{pair[0]}-{pair[1]}'] = top_level

    main_json_path = os.path.join(examples_dir, f'{args.name}_examples.json')
    with open(main_json_path, 'w') as f:
        json.dump(main_index, f)

def main(args):
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    if job_id == 'local':
        save_dir = args.out_path
    else:
        save_dir = os.path.join(f'/work/{job_id}', args.dataset_type, args.name)

    print(f"Saving to {save_dir}")

    print("Extracting pairs")
    worst_pairs_dict = get_worst_pairs(args)
    print("Extracting matches")
    get_matches(worst_pairs_dict, args)
    print("Exporting images")
    export_images(worst_pairs_dict, save_dir, args)
    print("Saving json")
    save_json(worst_pairs_dict, save_dir, args)

    tar_file_path = os.path.join(save_dir, f'{args.name}.tar.gz')
    print(f"Making tar archive: {tar_file_path}")
    with tarfile.open(tar_file_path, "w:gz") as tar:
        # in arcname include everything after 'mdrpbench'
        tar.add(save_dir, arcname=os.path.basename(save_dir))

    final_tar_path = os.path.join(args.out_path, f'{args.name}.tar.gz')


    print("Moving tar archive to: ", final_tar_path)
    shutil.move(tar_file_path, final_tar_path)

if __name__ == '__main__':
    args = parse_args()

    job_args = []

    if args.config_path is not None:
        for name, config, dataset_type in config_iterator(args.config_path, return_dataset_type=True):
            single_args = copy.copy(args)
            single_args.name = name
            single_args.dataset_type = dataset_type
            single_args.work_path = config["work_path"]
            single_args.data_path = config["path"]
            single_args.out_path = '/projects/p1358-25-2/mdrpbench_examples'

            job_args.append(single_args)

        executor = submitit.AutoExecutor(folder='/home/kocurvik/logs/submitit_logs/')
        executor.update_parameters(
            slurm_account=args.account,
            slurm_partition=args.queue,
            mem_gb=args.mem_gb,
            timeout_min=args.timeout_min,
            cpus_per_task=args.num_workers
        )

        jobs = executor.map_array(main, job_args)

    else:
        main(args)

