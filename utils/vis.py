import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from depth_estimators.infer_depth import ALL_MDEs

def get_n_colors(n):
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in range(n)]
    return [plt.cm.hsv(i / n) for i in range(n)]


def get_mde_basename_color_dict():
    basenames = [x.split('Calib')[0] for x in sorted(ALL_MDEs.keys())]
    colors = get_n_colors(len(basenames))
    return {base_name: colors[i] for i, base_name in enumerate(basenames)}


def plot_scatter_pose_depth(pose_df, depth_df, dataset, remove_outliers=False, out_dir='vis'):
    pose_df = pose_df[pose_df['dataset'] == dataset].copy()
    depth_df = depth_df[depth_df['dataset'] == dataset].copy()

    mde_list = depth_df['mde'].unique().tolist()

    if remove_outliers:
        mde_list = [x for x in mde_list if 'Infini' not in x and 'AnythingV2' not in x]
        if dataset == 'lamar':
            mde_list = [x for x in mde_list if 'DepthPro' not in x]

    depth_df = depth_df[depth_df['mde'].isin(mde_list)]
    base_names = list(set(x.split('-')[0].split('Calib')[0] for x in mde_list))
    color_dict = get_mde_basename_color_dict()

    depth_evals = ['A.Rel_si', 'd1_si', 'A.Rel_ssi', 'd1_ssi']

    n = len(depth_evals)
    ncols = 2
    nrows = 2

    all_solvers = pose_df['solver'].unique().tolist()
    solvers = [s for s in all_solvers if 'baseline' not in s]

    os.makedirs(out_dir, exist_ok=True)

    for solver in solvers:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(f'Dataset: {dataset}, Solver: {solver}')
        axes = axes.flatten()

        for idx, metric in enumerate(depth_evals):
            ax = axes[idx]
            for mde in mde_list:
                base_name = mde.split('-')[0].split('Calib')[0]
                if 'Calib' in mde and any(s in solver for s in ('vf', 'sf', 'mdecalib')):
                    continue
                depth_row = depth_df[depth_df['mde'] == mde]
                if depth_row.empty or metric not in depth_row.columns or pd.isna(depth_row[metric].iloc[0]):
                    continue
                depth_val = depth_row[metric].iloc[0]
                if 'd1' in metric:
                    depth_val = 100 - depth_val
                pose_rows = pose_df[(pose_df['mde'] == mde) & (pose_df['solver'] == solver)]
                if pose_rows.empty:
                    continue
                pose_vals = pose_rows['pose_mAA_10'].tolist()
                marker = 'o' if 'Calib' in mde else '*'
                ax.plot([depth_val] * len(pose_vals), pose_vals,
                        color=color_dict.get(base_name, 'black'), marker=marker, linestyle='dotted')
            ax.set_xlabel(f"Depth {metric}")
            ax.set_ylabel("Pose mAA(10)")

            baseline_solver = f'baseline_{solver.split("_shift")[0]}'
            if baseline_solver in all_solvers and '_ro' not in solver:
                gt_rows = pose_df[(pose_df['mde'] == 'gt') & (pose_df['solver'] == baseline_solver)]
                for pose_val in gt_rows['pose_mAA_10']:
                    ax.axhline(pose_val, linestyle='dashed', color='gray')

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        color_handles = [Line2D([0], [0], color=c, marker='s', linestyle='None', label=lbl)
                         for lbl, c in color_dict.items() if lbl in base_names]
        marker_handles = [Line2D([0], [0], color='gray', marker='*', linestyle='None', label='Normal'),
                          Line2D([0], [0], color='gray', marker='o', linestyle='None', label='Calib')]
        fig.legend(handles=color_handles + marker_handles, loc='center right')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        suffix = '_no_outliers' if remove_outliers else '_all'
        plt.savefig(os.path.join(out_dir, f'scatter_pose_depth_{dataset}_{solver}{suffix}.png'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_csv', type=str, default='csv_results/pose_results.csv')
    parser.add_argument('--depth_csv', type=str, default='csv_results/depth_results.csv')
    parser.add_argument('--out_dir', type=str, default='vis')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to plot (default: all datasets in the CSV)')
    parser.add_argument('--remove_outliers', action='store_true', default=False)
    args = parser.parse_args()

    pose_df = pd.read_csv(args.pose_csv)
    depth_df = pd.read_csv(args.depth_csv)

    datasets = [args.dataset] if args.dataset else pose_df['dataset'].unique().tolist()

    for dataset in datasets:
        plot_scatter_pose_depth(pose_df, depth_df, dataset,
                                remove_outliers=args.remove_outliers,
                                out_dir=args.out_dir)


if __name__ == '__main__':
    main()