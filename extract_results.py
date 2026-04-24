import copy
import json
import os

import pandas as pd

from utils.results import merge_summary_results, print_results_all, merge_summary_depth_results, \
    flatten_pose_metrics, flatten_depth_metrics


def parse_args():
    global args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('-st', '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-rt', '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to write unified JSON (default: <results_dir>/all.json)')
    parser.add_argument('-ed', '--eval_depth', action='store_true', default=False)
    parser.add_argument('--matches', type=str, default='splg_2048_noresize')
    parser.add_argument('--out_dir', type=str, default='csv_results')
    parser.add_argument('-a', '--append', action='store_true', default=False,
                        help='Append to existing CSV instead of overwriting the whole file')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='When used with --append, overwrite existing rows matching the key columns')
    args = parser.parse_args()
    return args


def save_csv(df, path, key_cols, append=False, overwrite=False, keep_slim_cols=None):
    if keep_slim_cols is not None:
        cols = [c for c in key_cols + keep_slim_cols if c in df.columns]
        df = df[cols].round(6)
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


def process_single_dataset(args):
    all_metrics = merge_summary_results(args)
    flat_pose_metrics = flatten_pose_metrics(all_metrics)

    if args.eval_depth:
        depth_metrics = merge_summary_depth_results(args)
        flat_depth_metrics = flatten_depth_metrics(depth_metrics)
    else:
        flat_depth_metrics = None

    return flat_pose_metrics, flat_depth_metrics


if __name__ == '__main__':
    args = parse_args()
    if args.config_path is not None:
        with open(args.config_path) as f:
            dataset_config = json.load(f)

        pose_dfs = []
        depth_dfs = []
        for name, config in dataset_config.items():
            single_args = copy.copy(args)
            single_args.name = name
            single_args.eval_depth = args.eval_depth and "contains_gt_depth" in config and config["contains_gt_depth"]
            single_args.data_path = config["work_path"]
            flat_pose_results, flat_depth_results = process_single_dataset(single_args)
            flat_pose_results.insert(0, 'dataset', name)
            pose_dfs.append(flat_pose_results)
            if flat_depth_results is not None:
                flat_depth_results.insert(0, 'dataset', name)
                depth_dfs.append(flat_depth_results)

        all_pose_df = pd.concat(pose_dfs, ignore_index=True)
        all_depth_df = pd.concat(depth_dfs, ignore_index=True) if depth_dfs else None
    else:
        all_pose_df, all_depth_df = process_single_dataset(args)
        all_pose_df.insert(0, 'dataset', args.name)
        if all_depth_df is not None:
            all_depth_df.insert(0, 'dataset', args.name)

    os.makedirs(args.out_dir, exist_ok=True)
    save_csv(all_pose_df, os.path.join(args.out_dir, 'pose_results.csv'),
             ['dataset', 'mde', 'iters', 'solver'], args.append, args.overwrite)
    save_csv(all_pose_df, os.path.join(args.out_dir, 'slim_pose_results.csv'),
             ['dataset', 'mde', 'iters', 'solver'], args.append, args.overwrite,
             keep_slim_cols=['pose_mAA_10', 'focal_mAA_10', 'median_pose_err', 'median_f_err', 'mean_mde_runtime', 'mean_runtime', 'mean_inliers'])
    if all_depth_df is not None:
        save_csv(all_depth_df, os.path.join(args.out_dir, 'depth_results.csv'), ['dataset', 'mde'], args.append, args.overwrite)




