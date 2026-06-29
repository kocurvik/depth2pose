import argparse
import os
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
from tqdm import tqdm

from depth_estimators.infer_depth import get_mde_model
from utils.geometry import R_err_fun, t_err_fun
from utils.results import save_summary_results, print_results_all
from utils.storage import save_full_results, get_full_results_h5_path, load_full_results


REGRESSED_POSE_MODELS = {
    'VGGT': ['VGGT-1B'],
    # 'Pi3': ['Pi3X'],
    'Pi3Calib': ['Pi3X'],
    # 'MapAnything': ['map-anything'],
    'MapAnythingCalib': ['map-anything'],
    # 'DepthAnythingV3': ['DA3-LARGE-1.1', 'DA3-GIANT-1.1'],
    'DepthAnythingV3Calib': ['DA3-LARGE-1.1', 'DA3-GIANT-1.1'],
}


def get_regressed_pose_model_name(model_name, weights):
    if model_name == 'VGGT':
        return f'VGGT-{weights}'
    if model_name in ('Pi3', 'Pi3Calib'):
        intrinsics = 'Calib' if 'Calib' in model_name else ''
        return f'Pi3{intrinsics}-{weights}'
    if model_name in ('MapAnything', 'MapAnythingCalib'):
        intrinsics = 'Calib' if 'Calib' in model_name else ''
        return f'MapAnything{intrinsics}-{weights}'
    if model_name in ('DepthAnythingV3', 'DepthAnythingV3Calib'):
        intrinsics = 'Calib' if 'Calib' in model_name else ''
        return f'DepthAnythingV3{intrinsics}-{weights}'
    raise NotImplementedError(f"Model {model_name} not implemented for regressed-pose evaluation")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st', '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-rt', '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('--variance', type=int, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--work_path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--matches', type=str, default='regressed_pose')
    parser.add_argument('dataset_path', nargs='?', default=None)
    return parser.parse_args()


def _as_w2c(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = extrinsic
        extrinsic = out
    if extrinsic.shape != (4, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 extrinsic, got {extrinsic.shape}")
    return extrinsic


def _relative_pose_from_w2c(extrinsic1, extrinsic2):
    extrinsic1 = _as_w2c(extrinsic1)
    extrinsic2 = _as_w2c(extrinsic2)
    R1, T1 = extrinsic1[:3, :3], extrinsic1[:3, 3]
    R2, T2 = extrinsic2[:3, :3], extrinsic2[:3, 3]
    R = R2 @ R1.T
    t = T2 - R @ T1

    # u, _, vh = np.linalg.svd(R)
    # R = u @ vh
    # if np.linalg.det(R) < 0:
    #     u[:, -1] *= -1
    #     R = u @ vh
    return R, t


def _make_result(out_dict, img_name_1, img_name_2, R_gt, t_gt, K1_gt, K2_gt):
    R_est, t_est = _relative_pose_from_w2c(out_dict['extrinsic1'], out_dict['extrinsic2'])
    K1_est = out_dict.get('K1')
    K2_est = out_dict.get('K2')
    if K1_est is None:
        K1_est = K1_gt
    if K2_est is None:
        K2_est = K2_gt

    f1 = (K1_est[0, 0] + K1_est[1, 1]) / 2
    f2 = (K2_est[0, 0] + K2_est[1, 1]) / 2
    f1_gt = (K1_gt[0, 0] + K1_gt[1, 1]) / 2
    f2_gt = (K2_gt[0, 0] + K2_gt[1, 1]) / 2
    f1_err = np.abs(f1 - f1_gt) / f1_gt
    f2_err = np.abs(f2 - f2_gt) / f2_gt

    result = {
        'experiment': 'regressed_pose',
        'iterations': 0,
        'image_name_1': img_name_1,
        'image_name_2': img_name_2,
        'R': R_est,
        'R_gt': R_gt,
        't': t_est,
        't_gt': t_gt,
        'f1': f1,
        'f2': f2,
        'f1_gt': f1_gt,
        'f2_gt': f2_gt,
        'R_err': R_err_fun(R_est, R_gt),
        't_err': t_err_fun(t_est, t_gt),
        'f1_err': f1_err,
        'f2_err': f2_err,
        'f_err': np.sqrt(f1_err * f2_err),
        'runtime': out_dict['runtime'],
        'info': {
            'inlier_ratio': 1.0,
            'model_score': 0.0,
            'iterations': 0,
            'num_inliers': 0,
            'refinements': 0,
            'inliers': np.array([], dtype=bool),
        },
    }
    return result


def eval_single_model(model, args):
    args.full_results_type = 'regression_full'
    print("Making dirs: ")
    print(os.path.join(args.work_path, 'regression_full_results'))
    print(os.path.join(args.work_path, 'summary_results'))
    os.makedirs(os.path.join(args.work_path, 'regression_full_results'), exist_ok=True)
    os.makedirs(os.path.join(args.work_path, 'summary_results'), exist_ok=True)

    args.depth = model.name
    if os.path.exists(get_full_results_h5_path(args)) and not args.recalc and not args.load:
        print(f"Results for {model.name} already available. Skipping.")
        return

    experiments = ['regressed_pose']
    if args.load:
        full_results = load_full_results(args)
        mde_runtimes = [r['runtime'] / 1e6 for r in full_results]
        save_summary_results(experiments, full_results, mde_runtimes, args)
        return

    print(f"Loading model {model.name}")
    model.load_model()

    name_path = os.path.join(args.work_path, args.name)
    image_pair_list_path = f'{name_path}_image_pairs.txt'
    with open(image_pair_list_path, 'r') as f:
        pair_list = [x.strip().split(',')[:2] for x in f.readlines()]
    if args.first is not None:
        pair_list = pair_list[:args.first]

    with h5py.File(f'{name_path}.h5', 'r') as f_images_h5:
        f_images = {key: np.array(f_images_h5[key]) for key in f_images_h5.keys()}

    full_results = []
    for img_name_1, img_name_2 in tqdm(pair_list):
        K1_gt = np.array(f_images[f'{img_name_1}_K'])
        R1 = np.array(f_images[f'{img_name_1}_R'])
        T1 = np.array(f_images[f'{img_name_1}_T'])
        K2_gt = np.array(f_images[f'{img_name_2}_K'])
        R2 = np.array(f_images[f'{img_name_2}_R'])
        T2 = np.array(f_images[f'{img_name_2}_T'])

        R_gt = R2 @ R1.T
        t_gt = T2 - R_gt @ T1

        img_path_1 = os.path.join(args.dataset_path, Path(PureWindowsPath(img_name_1)))
        img_path_2 = os.path.join(args.dataset_path, Path(PureWindowsPath(img_name_2)))

        size1 = f_images.get(f'{img_name_1}_size')
        size1_orig = f_images.get(f'{img_name_1}_size_orig')
        size2 = f_images.get(f'{img_name_2}_size')
        size2_orig = f_images.get(f'{img_name_2}_size_orig')
        infer_kwargs = {'K': K1_gt, 'K1': K1_gt, 'K2': K2_gt}
        if size1 is not None and size1_orig is not None and (size1 != size1_orig).any():
            infer_kwargs['size1'] = size1
        if size2 is not None and size2_orig is not None and (size2 != size2_orig).any():
            infer_kwargs['size2'] = size2

        out_dict = model.infer_pair(img_path_1, img_path_2, **infer_kwargs)
        full_results.append(_make_result(out_dict, img_name_1, img_name_2, R_gt, t_gt, K1_gt, K2_gt))

    save_full_results(args, full_results)
    mde_runtimes = [r['runtime'] / 1e6 for r in full_results]
    save_summary_results(experiments, full_results, mde_runtimes, args)


def get_models_to_run(args):
    if args.model_name is not None:
        return [get_mde_model(args.model_name, args.pretrained_weights)]

    models = []
    for model_name, weights_list in REGRESSED_POSE_MODELS.items():
        for weights in weights_list:
            models.append(get_mde_model(model_name, weights))
    return models


def main(args):
    for model in get_models_to_run(args):
        eval_single_model(model, args)
    print_results_all(args)


if __name__ == '__main__':
    args = parse_args()
    if args.config_path is not None:
        import copy
        from utils.config import config_iterator

        for name, config in config_iterator(args.config_path):
            single_args = copy.copy(args)
            single_args.name = name
            single_args.work_path = config['work_path']
            single_args.dataset_path = config.get('dataset_path', config['path'])
            main(single_args)
    else:
        main(args)
