"""Evaluate your own monocular depth estimator (MDE) on the D2P benchmark.

1. Implement the MyMDE class below (load_model, name and infer).
2. Run from the root of this repository, e.g.:

       python eval_my_mde.py                       # MDE predicts depth (and optionally K) on its own
       python eval_my_mde.py --use_gt_K            # MDE gets the ground truth intrinsics as input
       python eval_my_mde.py --matches splg --num_workers 16

The script:
  - downloads the D2P dataset from hugging face (https://huggingface.co/datasets/floodgab/d2p_dataset) into
    --data_path if it is not there yet,
  - runs your MDE on all benchmark images (the depth maps are stored next to the dataset and reused on the next run,
    use --recalc after changing your model). An interrupted run continues with the first scene that was not
    finished,
  - estimates the relative pose of all benchmark image pairs from the matches and your depth with 1000 RANSAC
    iterations using the solvers:
        calib        - GT intrinsics, depth used for the minimal solver and scoring (Sampson + reprojection error)
        calib_ro     - GT intrinsics, scoring only by the reprojection error
        mdecalib     - intrinsics predicted by the MDE (only if infer returns K and --use_gt_K is not set)
        mdecalib_ro  - as mdecalib, scoring only by the reprojection error
  - appends the results to csv_results/d2p_<matches>_slim_pose_results.csv (per scene) and
    csv_results/per_group_d2p_<matches>_slim_pose_results.csv (mean over the statues / vegetation scenes).
    To add your MDE to the results table, open a pull request with these two files.

The metric is the pose mAA (area under the recall curve of max(rotation error, translation direction error) up to
10 degrees). See the paper for details.
"""
import argparse
import json
import os
import re
from pathlib import PureWindowsPath
from time import perf_counter_ns

import h5py
import numpy as np
import pandas as pd
import torch

from depth_estimators.base import BaseDepthEstimator
from depth_estimators.infer_depth import get_depth_path, infer_depth_scene, is_depth_complete, read_image_list
from extract_results import collect_results, get_csv_paths, get_default_args as get_default_extract_args, save_results
from utils.config import config_iterator
from utils.poselib_check import warn_poselib
from utils.results import get_results_dir

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
HF_REPO_ID = 'floodgab/d2p_dataset'
MATCHES = {'loma': 'loma_2048_noresize', 'splg': 'splg_2048_noresize'}
RANSAC_ITERS = 1000


########################################################################################################################
# Your MDE
########################################################################################################################

class MyMDE(BaseDepthEstimator):
    """Implement your MDE here. See depth_estimators/*.py for the wrappers of the MDEs evaluated in the paper."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def load_model(self):
        """Load the network (weights, move to GPU, eval mode). Called once before the inference."""
        raise NotImplementedError('Implement MyMDE.load_model in eval_my_mde.py')

    @property
    def name(self):
        """Name of your MDE in the results table, e.g. 'MyMDE' or 'MyMDE-vitl' for a specific variant / backbone.
        Use letters, digits and '.', '+', '-' only. 'Calib' is added automatically when run with --use_gt_K
        (e.g. 'MyMDECalib-vitl')."""
        return 'MyMDE'

    def infer(self, image, K=None, size=None, **kwargs):
        """Predict depth for one image.

        Args:
            image: path to the RGB image.
            K: 3x3 ground truth intrinsic matrix in pixels. Only passed when the script is run with --use_gt_K.
            size: (width, height) to resize the image to before inference. Not used for D2P (always None).

        Returns:
            dict with
              'depth':   HxW numpy array with the depth (z coordinate, not the distance along the ray) for every pixel
                         at the full image resolution. The depth may be metric or known only up to scale,
                         invalid values should be <= 0, nan or inf.
              'runtime': inference time in nanoseconds (time.perf_counter_ns).
              'K':       (optional) 3x3 intrinsic matrix in pixels predicted by the MDE. If returned (and --use_gt_K
                         is not set) the poses are also estimated with the predicted intrinsics (mdecalib solvers).
        """
        raise NotImplementedError('Implement MyMDE.infer in eval_my_mde.py')

        # Example of the structure (see e.g. depth_estimators/MoGe.py for a full implementation):
        #
        # img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # x = torch.tensor(img / 255, dtype=torch.float32).permute(2, 0, 1)[None].cuda()
        # start_time = perf_counter_ns()
        # with torch.no_grad():
        #     depth, K_pred = self.model(x) if K is None else self.model(x, intrinsics=K)
        # torch.cuda.synchronize()
        # runtime = perf_counter_ns() - start_time
        # return {'depth': depth[0].cpu().numpy(), 'K': K_pred[0].cpu().numpy(), 'runtime': runtime}


########################################################################################################################
# Benchmark
########################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate your MDE (MyMDE in this file) on the D2P benchmark.')
    parser.add_argument('--data_path', type=str, default=os.path.join(REPO_ROOT, 'd2p'),
                        help='Where the D2P dataset is (or will be downloaded to). Depth maps and per scene results '
                             'are also stored here. Default: ./d2p in the repository.')
    parser.add_argument('--matches', type=str, default='loma',
                        help=f'Matches used for pose estimation: {", ".join(MATCHES)} (default: loma).')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of processes for pose estimation (default: 8).')
    parser.add_argument('--use_gt_K', action='store_true', default=False,
                        help='Pass the ground truth intrinsics to MyMDE.infer. The mdecalib solvers are not run.')
    parser.add_argument('--recalc', action='store_true', default=False,
                        help='Recompute depth and poses even if results from a previous run exist (use after changing '
                             'your MDE). Without it, an interrupted run continues with the first unfinished scene.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Replace the rows of an MDE with the same name already present in the results CSVs.')
    parser.add_argument('--out_dir', type=str, default=os.path.join(REPO_ROOT, 'csv_results'),
                        help='Directory with the results CSVs (default: csv_results in the repository).')
    parser.add_argument('--revision', type=str, default='main', help='Revision of the hugging face dataset.')
    args = parser.parse_args()
    args.matches = MATCHES.get(args.matches, args.matches)
    return args


def get_depth_name(model_name, use_gt_K):
    """Name of the MDE in the results, following the convention of the paper ('<Model>Calib-<variant>' when the GT
    intrinsics are used during inference)."""
    if not re.fullmatch(r'[A-Za-z0-9.+-]+', model_name):
        raise ValueError(f"Invalid MDE name '{model_name}'. Use letters, digits and '.', '+', '-' only.")
    base, sep, variant = model_name.partition('-')
    if not use_gt_K or base.endswith('Calib'):
        return model_name
    return f'{base}Calib{sep}{variant}'


def get_d2p_config(data_path):
    """D2P config (see datasets/d2p_hf.json) with the dataset located in data_path."""
    with open(os.path.join(REPO_ROOT, 'datasets', 'd2p_hf.json')) as f:
        config = json.load(f)
    for group, group_config in config.items():
        group_config['path'] = os.path.join(data_path, 'd2p_dataset', group)
        group_config['work_path'] = os.path.join(data_path, 'd2p_extracted', group)
    return config


def get_missing_files(config, matches):
    missing = []
    for name, scene_config in config_iterator(config):
        name_path = os.path.join(scene_config['work_path'], name)
        for suffix in ('.h5', '_image_list.txt', '_image_pairs.txt', f'_{matches}.h5'):
            if not os.path.exists(f'{name_path}{suffix}'):
                missing.append(f'{name_path}{suffix}')
        if os.path.exists(f'{name_path}_image_list.txt'):
            for image_name in read_image_list(scene_config['work_path'], name):
                image_path = os.path.join(scene_config['path'], PureWindowsPath(image_name).as_posix())
                if not os.path.exists(image_path):
                    missing.append(image_path)
    return missing


def ensure_dataset(config, args):
    missing = get_missing_files(config, args.matches)
    if not missing:
        return

    from huggingface_hub import snapshot_download

    print(f'{len(missing)} files of the D2P dataset are missing in {args.data_path} (e.g. {missing[0]}).')
    print(f'Downloading the D2P dataset from https://huggingface.co/datasets/{HF_REPO_ID} into {args.data_path}')
    snapshot_download(HF_REPO_ID, repo_type='dataset', revision=args.revision, local_dir=args.data_path,
                      allow_patterns=['d2p_dataset/*', 'd2p_extracted/*'], ignore_patterns=['*.DS_Store'])

    missing = get_missing_files(config, args.matches)
    if missing:
        raise RuntimeError(f'{len(missing)} files are still missing after the download, e.g. {missing[:5]}')


def has_mde_K(f_depth_path, image_list):
    with h5py.File(f_depth_path, 'r') as f:
        return all(f'{image_name}_K' in f for image_name in image_list)


def run_inference(model, depth_name, config, args):
    """Run the MDE on all scenes. Returns the list of scenes for which the depth was (re)computed and whether the MDE
    predicted the intrinsics for all images."""
    scenes = list(config_iterator(config))
    todo = [name for name, scene_config in scenes
            if args.recalc or not is_depth_complete(get_depth_path(scene_config['work_path'], name, depth_name))]

    if len(todo) < len(scenes):
        print(f'Reusing the depth maps of {depth_name} from a previous run for {len(scenes) - len(todo)} scenes. '
              f'Use --recalc if you changed your MDE.')

    if todo:
        print(f'Loading {depth_name}')
        model.load_model()

    for i, (name, scene_config) in enumerate(scenes):
        if name not in todo:
            continue
        print(f'[{i + 1}/{len(scenes)}] Depth inference for {name}')
        infer_depth_scene(model, scene_config['work_path'], name, scene_config['path'], depth_name=depth_name,
                          pass_K=args.use_gt_K, recalc=args.recalc, load_model=False)

    predicts_K = all(has_mde_K(get_depth_path(scene_config['work_path'], name, depth_name),
                               read_image_list(scene_config['work_path'], name))
                     for name, scene_config in scenes)
    return todo, predicts_K


def get_solvers(args, predicts_K):
    solvers = ['calib']
    if predicts_K and not args.use_gt_K:
        solvers.append('mdecalib')
    return solvers + [f'{x}_ro' for x in solvers]


def has_pose_results(eval_args, solvers):
    json_path = os.path.join(get_results_dir(eval_args), f'{eval_args.depth}.json')
    if not os.path.exists(json_path):
        return False
    with open(json_path) as f:
        metrics = json.load(f)
    return set(solvers) <= set(metrics.get(str(RANSAC_ITERS), {}).keys())


def run_pose_estimation(depth_name, solvers, recomputed_depth, config, args):
    # imported here as eval_pose needs PoseLib, which is checked in main() first
    from eval_pose import eval_single_mde, get_default_args as get_default_eval_args

    scenes = list(config_iterator(config))
    for i, (name, scene_config) in enumerate(scenes):
        eval_args = get_default_eval_args(work_path=scene_config['work_path'], name=name, matches=args.matches,
                                     depth=depth_name, explicit_solvers=','.join(solvers), max_iters_only=True,
                                     num_workers=args.num_workers)
        assert eval_args.max_iters_only and RANSAC_ITERS == 1000  # eval_pose then only runs 1000 iterations

        if not args.recalc and name not in recomputed_depth and has_pose_results(eval_args, solvers):
            print(f'[{i + 1}/{len(scenes)}] Pose results for {name} available from a previous run. Skipping.')
            continue

        print(f'[{i + 1}/{len(scenes)}] Pose estimation for {name}')
        eval_single_mde(eval_args)


def save_to_csv(depth_name, solvers, config, args):
    extract_args = get_default_extract_args(config_path=config, prefix='d2p', matches=args.matches,
                                            out_dir=args.out_dir, append=True, overwrite=True)
    pose_df, _ = collect_results(extract_args, mde_list=[depth_name])
    pose_df = pose_df[(pose_df['iters'] == RANSAC_ITERS) & pose_df['solver'].isin(solvers)]

    n_scenes = len(list(config_iterator(config)))
    if pose_df['dataset'].nunique() != n_scenes:
        raise RuntimeError(f'Results found only for {pose_df["dataset"].nunique()} of {n_scenes} scenes.')

    save_results(extract_args, pose_df, slim_only=True)
    _, slim_path, per_group_path = get_csv_paths(extract_args)
    print(f'Results of {depth_name} saved to {slim_path} and {per_group_path}')
    return per_group_path


def check_existing_rows(depth_name, args):
    extract_args = get_default_extract_args(prefix='d2p', matches=args.matches, out_dir=args.out_dir)
    for path in get_csv_paths(extract_args)[1:]:
        if os.path.exists(path) and depth_name in set(pd.read_csv(path)['mde']):
            if not args.overwrite:
                raise SystemExit(f"Results of an MDE named '{depth_name}' are already in {path}. Change the name of "
                                 f"your MDE (its name property), run with --overwrite to replace them or use --out_dir "
                                 f"to write the results elsewhere.")
            print(f"Results of '{depth_name}' in {path} will be replaced.")


def print_summary(depth_name, solvers, per_group_path):
    df = pd.read_csv(per_group_path)
    df = df[(df['mde'] == depth_name) & (df['iters'] == RANSAC_ITERS) & df['solver'].isin(solvers)]
    table = df.pivot(index='solver', columns='dataset', values='pose_mAA_10')
    table['mean'] = table.mean(axis=1)
    print(f'\nPose mAA(10) of {depth_name} with {RANSAC_ITERS} RANSAC iterations:')
    print(table.round(2).to_string())
    print(f'Mean MDE runtime: {df["mean_mde_runtime"].mean():.1f} ms')


def main(args, model=None):
    # check before downloading and inference, pose estimation would fail only after the inference otherwise
    if not warn_poselib():
        raise SystemExit(1)

    model = MyMDE() if model is None else model
    model.requires_intrinsics = args.use_gt_K
    depth_name = get_depth_name(model.name, args.use_gt_K)
    print(f'Evaluating {depth_name} on D2P with {args.matches} matches')

    check_existing_rows(depth_name, args)

    config = get_d2p_config(args.data_path)
    ensure_dataset(config, args)

    recomputed_depth, predicts_K = run_inference(model, depth_name, config, args)
    solvers = get_solvers(args, predicts_K)
    if not predicts_K and not args.use_gt_K:
        print('The MDE does not return K, skipping the mdecalib solvers.')
    print(f'Solvers: {solvers}')

    run_pose_estimation(depth_name, solvers, recomputed_depth, config, args)
    per_group_path = save_to_csv(depth_name, solvers, config, args)
    print_summary(depth_name, solvers, per_group_path)


if __name__ == '__main__':
    main(parse_args())
