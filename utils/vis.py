import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from utils.tables import estimator_name

plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX to render text
    "font.family": "serif",        # Use serif for most text
    "font.serif": ["Times"],       # Specifically Times New Roman
    # "font.size": 10,               # Matches NeurIPS \normalsize
    # "axes.labelsize": 10,
    # "legend.fontsize": 8,          # Matches \footnotesize
    # "xtick.labelsize": 8,
    # "ytick.labelsize": 8,
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{times}
        % Add your custom commands here
        \newcommand{\mAA}{mAA(10$^\circ$)}
        \newcommand{\M}[1]{\mathbf{#1}}
                
        \newcommand{\eth}{ETH3D}
        \newcommand{\sintel}{Sintel}
        \newcommand{\scannetpp}{ScanNet++}
        \newcommand{\lamar}{Lamar}
        
        \newcommand{\baselinecalib}{B}
        \newcommand{\baselinecalibshift}{B$_f$}
        \newcommand{\calib}{D}
        \newcommand{\calibshift}{D$_s$}
        \newcommand{\mysf}{D$_{f}$}
        \newcommand{\sfshift}{D$_{s,f}$}
        
        \newcommand{\mdecalib}{MK}        
        \newcommand{\mdecalibshift}{MK$_s$}
        
        \newcommand{\mdecalibro}{MKR}        
        \newcommand{\mdecalibshiftro}{MKR$_s$}
        
        \newcommand{\calibro}{R}
        \newcommand{\calibshiftro}{R$_s$}
        \newcommand{\sfro}{R$_{f}$}
        \newcommand{\sfshiftro}{R$_{s,f}$}
    """,
    "pgf.rcfonts": False,          # Don't setup fonts from rc parameters
})


from depth_estimators.infer_depth import ALL_MDEs
from utils.results import get_mde_basename, get_backbone_name


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


def get_mde_marker(mde, basename=None, backbone=None):
    if basename is None:
        basename = get_mde_basename(mde)
    if backbone is None:
        backbone = get_backbone_name(mde)
    calib = 'Calib' in mde

    style = {}
    if calib:
        style['markeredgecolor'] = 'black'
    else:
        style['markeredgecolor'] = 'gray'

    if backbone == 'ViT-S':
        style['marker'] = 'o'
    elif backbone == 'ViT-B':
        style['marker'] = 's'
    elif backbone == 'ViT-G':
        style['marker'] = 'd'
    elif backbone == 'ConvNext':
        style['marker'] = 'h'
    else:
        style['marker'] = 'p'

    if 'MoGe' in basename:
        style['markerfacecolor'] = '#2ca02c'   # tab green
        if '1' in basename:
            style['markerfacecolor'] = '#98df8a'  # light green

    if 'UniDepth' in basename or 'UniK3D' in basename:
        style['markerfacecolor'] = '#1f77b4'   # tab blue
        if '1' in basename:
            style['markerfacecolor'] = '#aec7e8'  # light blue
        if '3' in basename:
            style['markerfacecolor'] = '#6baed6'  # medium blue (UniK3D)

    if 'Metric3D' in basename:
        style['markerfacecolor'] = '#17becf'   # tab teal

    if 'DepthAnything' in basename:
        style['markerfacecolor'] = '#ff9896'  # light red

    if 'DAv3' in basename:
        style['markerfacecolor'] = '#ff7f7f'   # tab red

        if 'Metric' in mde:
            style['markerfacecolor'] = '#ff4d4d'  # tab orange

    if 'VGGT' == basename:
        style['markerfacecolor'] = '#9467bd'   # tab purple

    if 'Pi3' == basename:
        style['markerfacecolor'] = '#ff7f0e'   # tab orange

    if 'MapAnything' in basename:
        style['markerfacecolor'] = '#e377c2'   # tab pink

    if 'DepthPro' == basename:
        style['markerfacecolor'] = '#8c564b'   # tab brown

    # Unused
    if 'InfiniDepth' == basename:
        style['markerfacecolor'] = '#7f7f7f'   # tab gray

    if 'markerfacecoloralt' in style:
        style['fillstyle'] = 'left'
    else:
        style['fillstyle'] = 'full'

    style['color'] = style['markerfacecolor']
    style['markersize'] = 8

    return style


def generate_legend_markers(out_dir='vis/legend'):
    os.makedirs(out_dir, exist_ok=True)

    styles = {}

    for k, v in ALL_MDEs.items():
        for b in v:
            mde = f'{k}-{b}'

            if 'Calib' in mde:
                continue

            basename = get_mde_basename(mde)

            backbones = [get_backbone_name(x) for x in ALL_MDEs[k]]
            if 'ViT-L' in backbones:
                backbone = 'ViT-L'
            else:
                backbone = backbones[0]

            style = get_mde_marker(basename, backbone=backbone)

            marker_style = {k: v for k, v in style.items() if k != 'color'}

            styles[basename] = marker_style

    print(list(styles.keys()))

    calib_style = get_mde_marker('UniK3DCalib-vitl')
    calib_style['markerfacecolor'] = 'white'
    styles['Calib'] = calib_style

    uncal_style = get_mde_marker('UniK3D-vitl')
    uncal_style['markerfacecolor'] = 'white'
    styles['NoCalib'] = uncal_style

    backbones = ['ViT-L', 'ViT-S', 'ViT-B', 'ViT-G', 'ConvNext']

    for backbone in backbones:
        style = get_mde_marker('UniK3D-vitl', backbone=backbone)
        style['markerfacecolor'] = 'gray'
        style['color'] = None
        styles[backbone] = style

    for basename, marker_style in styles.items():
        marker_style['markersize'] = 14
        fig, ax = plt.subplots(figsize=(0.4, 0.4))
        ax.plot([0.5], [0.5], linestyle='none', **marker_style)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.patch.set_alpha(0)
        plt.savefig(os.path.join(out_dir, f'{basename}.pdf'), #bbox_inches='tight', pad_inches=0.00,
                    transparent=True)
        plt.close(fig)








def plot_scatter_pose_depth(pose_df, depth_df, dataset, metric, remove_outliers=False, iters=None, out_dir='vis',
                            label_font_size=18, tick_font_size=14):
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
        mde_list = [x for x in mde_list if 'Infini' not in x and 'AnythingV2' not in x and 'DA3MONO' not in x]
        if dataset == 'lamar' or dataset == 'mean':
            mde_list = [x for x in mde_list if 'DepthPro' not in x]

    depth_df = depth_df[depth_df['mde'].isin(mde_list)]

    # depth_evals = ['A.Rel_si', 'd1_si', 'A.Rel_ssi', 'd1_ssi']
    # n = len(depth_evals)
    # ncols = 2
    # nrows = 2

    all_solvers = pose_df['solver'].unique().tolist()
    solvers = [s for s in all_solvers if 'baseline' not in s]

    if 'ssi' in metric:
        solvers = [s for s in solvers if 'shift' in s]
    else:
        solvers = [s for s in solvers if 'shift' not in s]

    os.makedirs(out_dir, exist_ok=True)

    for solver in solvers:
        # fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
        # fig.suptitle(f'Dataset: {dataset}, Solver: {solver}')
        # axes = axes.flatten()

        fig = plt.figure(figsize=[4, 3.5])

        for mde in mde_list:
            if 'Calib' in mde and any(s in solver for s in ('vf', 'sf', 'mdecalib')):
                continue
            depth_row = depth_df[depth_df['mde'] == mde]
            if depth_row.empty or metric not in depth_row.columns or pd.isna(depth_row[metric].iloc[0]):
                continue
            depth_val = depth_row[metric].iloc[0]
            pose_rows = pose_df[(pose_df['mde'] == mde) & (pose_df['solver'] == solver)]
            if pose_rows.empty:
                continue
            pose_vals = pose_rows['pose_mAA_10'].tolist()
            style = get_mde_marker(mde)

            plt.plot([depth_val] * len(pose_vals), pose_vals,
                    linestyle='dotted', **style, label=mde)

        if 'd' in metric:
            metric_name = '$\\delta_{1}$'
        else:
            metric_name = 'Rel'

        if 'ssi' in metric:
            metric_name += ' (affine inv.)'
        else:
            metric_name += ' (scale inv.)'
        plt.xlabel(metric_name, fontsize=label_font_size)
        plt.ylabel(f"{estimator_name(solver)} -- \mAA", fontsize=label_font_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_font_size)

        if 'd' in metric:
            plt.gca().invert_xaxis()

        baseline_solver = f'baseline_{solver.split("_shift")[0]}'
        if baseline_solver in all_solvers and '_ro' not in solver:
            gt_rows = pose_df[(pose_df['mde'] == 'none') & (pose_df['solver'] == baseline_solver)]
            for pose_val in gt_rows['pose_mAA_10']:
                plt.axhline(pose_val, linestyle='dashed', color='gray')

        # color_handles = [Line2D([0], [0], color=c, marker='s', linestyle='None', label=lbl)
        #                  for lbl, c in color_dict.items() if lbl in base_names]
        # marker_handles = [Line2D([0], [0], color='gray', marker='*', linestyle='None', label='Normal'),
        #                   Line2D([0], [0], color='gray', marker='o', linestyle='None', label='Calib')]
        # fig.legend(handles=color_handles + marker_handles, loc='center right')
        # plt.legend()
        plt.tight_layout()
        suffix = 'no_outliers' if remove_outliers else 'all'
        suffix += f'_{iters}' if iters is not None else ''

        plt.savefig(os.path.join(out_dir, f'scatter_pose_depth_{dataset}_{solver}_{metric}_{suffix}.pdf'))
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

    generate_legend_markers()

    metrics = ['A.Rel_si', 'd1_si', 'A.Rel_ssi', 'd1_ssi']

    for metric in metrics:
        plot_scatter_pose_depth(pose_df, depth_df, 'mean', metric,
                                remove_outliers=False,
                                out_dir=args.out_dir, iters=1000)

        plot_scatter_pose_depth(pose_df, depth_df, 'mean', metric,
                                remove_outliers=True,
                                out_dir=args.out_dir, iters=1000)


if __name__ == '__main__':
    main()