import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from depth_estimators.infer_depth import ALL_MDEs
from utils.results import get_mde_basename
from utils.tables import get_backbone_name


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


def get_mde_marker(mde):
    basename = get_mde_basename(mde)
    backbone = get_backbone_name(mde)
    calib = 'Calib' in mde

    style = {}
    if calib:
        style['markeredgecolor'] = 'black'

    if backbone == 'ViT-S':
        style['marker'] = 'o'
    elif backbone == 'ViT-B':
        style['marker'] = 's'
    elif backbone == 'ViT-G':
        style['marker'] = 'h'
    elif backbone == 'ConvNext':
        style['marker'] = 'D'
    else:
        style['marker'] = 'p'

    if 'MoGe' in basename:
        style['markerfacecolor'] = 'darkgreen'
        if '1' in basename:
            style['markerfacecoloralt'] = 'limegreen'

    if 'UniDepth' in basename or 'UniK3D' in basename:
        style['markerfacecolor'] = 'darkblue'
        if '1' in basename:
            style['markerfacecoloralt'] = 'lightsteelblue'
        if '3' in basename:
            style['markerfacecoloralt'] = 'royalblue'

    if 'Metric3D' in basename:
        style['markerfacecolor'] = 'turquoise'

    if 'DepthAnything' in basename:
        style['markerfacecolor'] = 'firebrick'
        if '2' in basename:
            style['markerfacecoloralt'] = 'lightsalmon'

    if 'VGGT' == basename:
        style['markerfacecolor'] = 'indigo'

    if 'Pi3' == basename:
        style['markerfacecolor'] = 'darkorange'

    if 'MapAnything' in basename:
        style['markerfacecolor'] = 'gold'

    if 'DepthPro' == basename:
        style['markerfacecolor'] = 'black'

    if 'InfiniDepth' == basename:
        style['markerfacecolor'] = 'slategray'

    if 'markerfacecoloralt' in style:
        style['fillstyle'] = 'left'
    else:
        style['fillstyle'] = 'full'

    style['color'] = style['markerfacecolor']
    style['markersize'] = 10

    return style


def plot_scatter_pose_depth(pose_df, depth_df, dataset, remove_outliers=False, iters=None, out_dir='vis'):
    if iters is not None:
        pose_df = pose_df[pose_df['iters'] == iters].copy()


    if dataset == 'mean':
        pose_df = pose_df.groupby(['mde', 'solver', 'iters'], as_index=False).mean(numeric_only=True)
        # pose_df = pose_df.groupby(['mde', 'solver', 'iters'])['pose_mAA_10'].mean().reset_index()
        depth_df = depth_df.groupby(['mde'], as_index=False).mean(numeric_only=True)
        pose_df['dataset'] = 'mean'
        depth_df['dataset'] = 'mean'
    else:
        pose_df = pose_df[pose_df['dataset'] == dataset].copy()
        depth_df = depth_df[depth_df['dataset'] == dataset].copy()

    mde_list = depth_df['mde'].unique().tolist()

    if remove_outliers:
        mde_list = [x for x in mde_list if 'Infini' not in x and 'AnythingV2' not in x]
        if dataset == 'lamar' or dataset == 'mean':
            mde_list = [x for x in mde_list if 'DepthPro' not in x]

    depth_df = depth_df[depth_df['mde'].isin(mde_list)]

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
                base_name = get_mde_basename(mde)
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
                style = get_mde_marker(mde)

                ax.plot([depth_val] * len(pose_vals), pose_vals,
                        linestyle='dotted', **style)
            ax.set_xlabel(f"Depth {metric}")
            ax.set_ylabel("Pose mAA(10)")

            baseline_solver = f'baseline_{solver.split("_shift")[0]}'
            if baseline_solver in all_solvers and '_ro' not in solver:
                gt_rows = pose_df[(pose_df['mde'] == 'gt') & (pose_df['solver'] == baseline_solver)]
                for pose_val in gt_rows['pose_mAA_10']:
                    ax.axhline(pose_val, linestyle='dashed', color='gray')

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        # color_handles = [Line2D([0], [0], color=c, marker='s', linestyle='None', label=lbl)
        #                  for lbl, c in color_dict.items() if lbl in base_names]
        # marker_handles = [Line2D([0], [0], color='gray', marker='*', linestyle='None', label='Normal'),
        #                   Line2D([0], [0], color='gray', marker='o', linestyle='None', label='Calib')]
        # fig.legend(handles=color_handles + marker_handles, loc='center right')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        suffix = '_no_outliers' if remove_outliers else '_all'
        suffix += f'_{iters}' if iters is not None else ''
        plt.savefig(os.path.join(out_dir, f'scatter_pose_depth_{dataset}_{solver}{suffix}.png'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_csv', type=str, default='csv_results/standard_pose_results.csv')
    parser.add_argument('--depth_csv', type=str, default='csv_results/standard_depth_results.csv')
    parser.add_argument('--out_dir', type=str, default='vis')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to plot (default: all datasets in the CSV)')
    parser.add_argument('--remove_outliers', action='store_true', default=False)
    args = parser.parse_args()

    pose_df = pd.read_csv(args.pose_csv)
    depth_df = pd.read_csv(args.depth_csv)

    datasets = [args.dataset] if args.dataset else pose_df['dataset'].unique().tolist()


    plot_scatter_pose_depth(pose_df, depth_df, 'mean',
                            remove_outliers=args.remove_outliers,
                            out_dir=args.out_dir, iters=1000)


if __name__ == '__main__':
    main()