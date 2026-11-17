import copy
import json
import os

import pandas as pd

from utils.config import config_iterator
from utils.results import merge_summary_results, print_results_all, merge_summary_depth_results, \
    flatten_pose_metrics, flatten_depth_metrics, get_results_dir


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_path', type=str, default='data')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('-st', '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-rt', '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('-ed', '--eval_depth', action='store_true', default=False)
    parser.add_argument('--matches', type=str, default='splg_2048_noresize')
    parser.add_argument('--out_dir', type=str, default='csv_results')
    parser.add_argument('--variance', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False,
                        help='Append to existing CSV instead of overwriting the whole file')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='When used with --append, overwrite existing rows matching the key columns')
    parser.add_argument('-r', '--regressed', action='store_true', default=False,
                        help='Extract results produced by (spawn_)eval_regressed_pose(_jobs).py')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.regressed:
        if args.variance:
            parser.error('--variance is not supported with --regressed')
        args.matches = 'regressed_pose'
    return args


def get_default_args(**overrides):
    """Namespace with the command line defaults of this script, updated with overrides. config_path may also be
    a config dict (see utils.config.config_iterator)."""
    args = parse_args([])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise AttributeError(f'Unknown extract_results argument: {k}')
        setattr(args, k, v)
    return args


def save_csv(df, path, key_cols, append=False, overwrite=False, keep_slim_cols=None, mean_over_groups=False, round=4):
    if keep_slim_cols is not None:
        cols = [c for c in key_cols + keep_slim_cols if c in df.columns]
        df = df[cols].round(2)
    else:
        df = df.round(round)

    if mean_over_groups:
        group_cols = [c for c in key_cols if c != 'dataset']
        df = df.groupby(group_cols)[keep_slim_cols].mean().reset_index()
        df.insert(df.columns.get_loc('group') + 1, 'dataset', df['group'])
        df = df.round(2)
        df.drop("group", axis=1, inplace=True)
        # the group name is now in the dataset column
        key_cols = [c for c in key_cols if c != 'group']

    if append and os.path.exists(path):
        existing = pd.read_csv(path)
        if overwrite:
            keep = existing.merge(df[key_cols].drop_duplicates(), on=key_cols, how='left', indicator=True)
            keep = keep[keep['_merge'] == 'left_only'].drop(columns='_merge')
            combined = pd.concat([keep, df], ignore_index=True)
        else:
            new_rows = df.merge(existing[key_cols].drop_duplicates(), on=key_cols, how='left', indicator=True)
            new_rows = new_rows[new_rows['_merge'] == 'left_only'].drop(columns='_merge')
            combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def process_single_dataset(args, mde_list=None):
    all_metrics = merge_summary_results(args)
    if mde_list is not None:
        all_metrics = {k: v for k, v in all_metrics.items() if k in mde_list}
    flat_pose_metrics = flatten_pose_metrics(all_metrics, variance=args.variance)
    if args.regressed:
        # regressed pose has no RANSAC, the inlier ratio is only a placeholder
        flat_pose_metrics = flat_pose_metrics.drop(columns='mean_inliers', errors='ignore')
    elif 'mean_inliers' in flat_pose_metrics.columns:
        flat_pose_metrics['mean_inliers'] *= 100

    if args.eval_depth:
        depth_metrics = merge_summary_depth_results(args)
        flat_depth_metrics = flatten_depth_metrics(depth_metrics)
    else:
        flat_depth_metrics = None

    return flat_pose_metrics, flat_depth_metrics


def collect_results(args, mde_list=None):
    """Collect the summary results of all scenes in args.config_path into data frames.

    Returns (all_pose_df, all_depth_df); all_depth_df is None unless args.eval_depth. If mde_list is given only
    results of these MDEs are collected.
    """
    if args.config_path is None:
        raise NotImplementedError('--config_path is required')

    pose_dfs = []
    depth_dfs = []

    for name, config, scene_group in config_iterator(args.config_path, return_dataset_type=True):

        single_args = copy.copy(args)
        single_args.name = name
        single_args.eval_depth = args.eval_depth #and "contains_gt_depth" in config and config["contains_gt_depth"]
        single_args.work_path = config["work_path"]
        if not os.path.isdir(get_results_dir(single_args)):
            print(f"No summary results found for {name} at {get_results_dir(single_args)}. Skipping.")
            continue
        flat_pose_results, flat_depth_results = process_single_dataset(single_args, mde_list=mde_list)
        flat_pose_results.insert(0, 'dataset', name)
        flat_pose_results.insert(0, 'group', scene_group)
        pose_dfs.append(flat_pose_results)
        if flat_depth_results is not None:
            flat_depth_results.insert(0, 'dataset', name)
            flat_depth_results.insert(0, 'group', scene_group)
            depth_dfs.append(flat_depth_results)

    all_pose_df = pd.concat(pose_dfs, ignore_index=True)
    all_depth_df = pd.concat(depth_dfs, ignore_index=True) if depth_dfs else None
    return all_pose_df, all_depth_df


def get_prefix(args):
    prefix = args.prefix
    if prefix is None:
        prefix = os.path.basename(args.config_path).split('_')[0]
    if args.variance:
        prefix = f'variance_{prefix}'
    return prefix


def get_keep_slim_cols(args):
    if args.variance:
        return ['pose_mAA_10', 'mean_inliers']
    elif args.regressed:
        return ['pose_mAA_10', 'f_mAA_10', 'mean_mde_runtime']
    else:
        return ['pose_mAA_10', 'mean_mde_runtime', 'mean_inliers']


def get_csv_paths(args):
    """Paths of the (full, slim, per group slim) pose results CSVs."""
    prefix = get_prefix(args)
    matches = args.matches.split('_')[0]
    return (os.path.join(args.out_dir, f'{prefix}_{matches}_pose_results.csv'),
            os.path.join(args.out_dir, f'{prefix}_{matches}_slim_pose_results.csv'),
            os.path.join(args.out_dir, f'per_group_{prefix}_{matches}_slim_pose_results.csv'))


def save_results(args, all_pose_df, all_depth_df=None, slim_only=False):
    """Save the collected results as CSVs into args.out_dir. With slim_only only the slim per scene and per group
    pose CSVs (the ones tracked in the repository) are written."""
    keep_slim_cols = get_keep_slim_cols(args)
    top_level_name = 'seed' if args.variance else 'iters'
    key_cols = ['group', 'dataset', 'mde', top_level_name, 'solver']
    full_path, slim_path, per_group_path = get_csv_paths(args)

    os.makedirs(args.out_dir, exist_ok=True)
    if not slim_only:
        save_csv(all_pose_df, full_path, key_cols, args.append, args.overwrite)
    save_csv(all_pose_df, slim_path, key_cols, args.append, args.overwrite, keep_slim_cols=keep_slim_cols)
    save_csv(all_pose_df, per_group_path, key_cols, args.append, args.overwrite,
             keep_slim_cols=keep_slim_cols, mean_over_groups=True)
    if all_depth_df is not None and not slim_only:
        save_csv(all_depth_df, os.path.join(args.out_dir, f'{get_prefix(args)}_depth_results.csv'),
                 ['group', 'dataset', 'mde'], args.append, args.overwrite)


if __name__ == '__main__':
    args = parse_args()
    all_pose_df, all_depth_df = collect_results(args)
    save_results(args, all_pose_df, all_depth_df)
