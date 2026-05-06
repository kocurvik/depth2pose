import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, linregress
import matplotlib.ticker as ticker

from utils.tables import estimator_name, DEPTH_METRIC_MAP

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
        \newcommand{\dsi}{$\delta_{1}^{si}$}
        \newcommand{\dssi}{$\delta_{1}^{ai}$}
        \newcommand{\relsi}{Rel$^{si}$}
        \newcommand{\relssi}{Rel$^{ai}$}
        \newcommand{\M}[1]{\mathbf{#1}}
                
        \newcommand{\eth}{ETH3D}
        \newcommand{\sintel}{Sintel}
        \newcommand{\scannetpp}{ScanNet++}
        \newcommand{\lamar}{Lamar}
        
        \newcommand{\baselinecalib}{$\mathcal{B}$}
        \newcommand{\baselinecalibshift}{$\mathcal{B}_{\mathrm{f}}$}
        \newcommand{\calib}{$\mathcal{D}$}
        \newcommand{\calibshift}{$\mathcal{D}_{\mathrm{a}}$}
        \newcommand{\mysf}{$\mathcal{D}_{\mathrm{f}}$}
        \newcommand{\sfshift}{$\mathcal{D}_{\mathrm{a,f}}$}
        
        \newcommand{\mdecalib}{$\mathcal{K}$}
        \newcommand{\mdecalibshift}{$\mathcal{K}_{\mathrm{a}}$}
        
        \newcommand{\calibro}{$\mathcal{R}$}
        \newcommand{\calibshiftro}{$\mathcal{R}_{\mathrm{a}}$}
        \newcommand{\sfro}{$\mathcal{R}_{\mathrm{f}}$}
        \newcommand{\sfshiftro}{$\mathcal{R}_{\mathrm{a,f}}$}
        
        \newcommand{\mdecalibro}{$\mathcal{KR}$}
        \newcommand{\mdecalibshiftro}{$\mathcal{KR}_{\mathrm{a}}$}

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

    if backbone == 'ViT-S':
        style['marker'] = 'o'
    elif backbone == 'ViT-B':
        style['marker'] = 'v'
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

            # backbones = [get_backbone_name(x) for x in ALL_MDEs[k]]
            # if 'ViT-L' in backbones:
            #     backbone = 'ViT-L'
            # else:
            #     backbone = backbones[0]

            # style = get_mde_marker(basename, backbone=backbone)
            style = get_mde_marker(basename)

            marker_style = {k: v for k, v in style.items()}
            marker_style['marker'] = 's'
            marker_style['markeredgecolor'] = None



            styles[basename] = marker_style


    print(list(styles.keys()))

    calib_style = get_mde_marker('UniK3DCalib-vitl')
    calib_style['markerfacecolor'] = 'gray'
    calib_style['marker'] = 's'
    styles['Calib'] = calib_style

    uncal_style = get_mde_marker('UniK3D-vitl')
    uncal_style['markerfacecolor'] = 'gray'
    uncal_style['color'] = 'gray'
    uncal_style['marker'] = 's'
    styles['NoCalib'] = uncal_style

    backbones = ['ViT-L', 'ViT-S', 'ViT-B', 'ViT-G', 'ConvNext']

    for backbone in backbones:
        style = get_mde_marker('UniK3D-vitl', backbone=backbone)
        style['markerfacecolor'] = 'gray'
        style['color'] = 'gray'
        styles[backbone] = style

    for basename, marker_style in styles.items():
        marker_style['markersize'] = 8
        fig, ax = plt.subplots(figsize=(0.4*8/14, 0.4*8/14))
        ax.plot([0.5], [0.5], linestyle='none', **marker_style)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.patch.set_alpha(0)
        plt.savefig(os.path.join(out_dir, f'{basename}.pdf'), #bbox_inches='tight', pad_inches=0.00,
                    transparent=True)
        plt.close(fig)








def plot_scatter_pose_depth(pose_df, depth_df, dataset, metric, remove_outliers=False, iters=None, out_dir='vis',
                            label_font_size=18, tick_font_size=14, matches='splg', solvers=None, figsize=None):
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
        plot_mde_list = [x for x in mde_list if 'Infini' not in x and 'AnythingV2' not in x and 'DA3MONO' not in x]
        if dataset == 'lamar' or dataset == 'mean':
            plot_mde_list = [x for x in plot_mde_list if 'DepthPro' not in x]

    depth_df = depth_df[depth_df['mde'].isin(mde_list)]

    if solvers is None:
        all_solvers = pose_df['solver'].unique().tolist()
        solvers = [s for s in all_solvers if 'baseline' not in s]

    if 'ssi' in metric:
        solvers = [s for s in solvers if 'shift' in s]
    else:
        solvers = [s for s in solvers if 'shift' not in s]

    os.makedirs(out_dir, exist_ok=True)

    for solver in solvers:
        if figsize is None:
            figsize = [4 * 0.8, 3.5 * 0.8]

        fig = plt.figure(figsize=figsize)

        all_x = []
        all_y = []
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

            xs = [depth_val] * len(pose_vals)
            if (not remove_outliers) or (mde in plot_mde_list):
                plt.plot(xs, pose_vals,
                        linestyle='dotted', **style, label=mde)
            all_x.extend(xs)
            all_y.extend(pose_vals)

        suffix = 'no_outliers' if remove_outliers else 'all'
        suffix += f'_{iters}' if iters is not None else ''
        pdf_path = f'scatter_pose_depth_{matches}_{dataset}_{solver}_{metric}_{suffix}.pdf'

        if len(all_x) > 1:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            mask = ~np.isnan(all_x) & ~np.isnan(all_y)
            all_x, all_y = all_x[mask], all_y[mask]
            if len(all_x) > 1:
                corr, _ = pearsonr(all_x, all_y)
                slope, intercept, _, _, _ = linregress(all_x, all_y)
                xlim = plt.gca().get_xlim()
                x_range = np.array(xlim)
                plt.plot(x_range, slope * x_range + intercept, color='gray', linestyle='--', zorder=0)
                plt.gca().set_xlim(xlim)
                plt.text(0.05, 0.95, f'$r={corr:.4f}$', transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=tick_font_size)
                print(f"{pdf_path}: {corr:.4f}")

        plt.xlabel(DEPTH_METRIC_MAP[metric], fontsize=label_font_size)
        plt.ylabel(f"{estimator_name(solver)} -- \mAA", fontsize=label_font_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator('auto', integer=True))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator('auto', integer=True))

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, pdf_path))
        plt.close(fig)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pose_csv', type=str, default='csv_results/standard_splg_pose_results.csv')
    # parser.add_argument('--depth_csv', type=str, default='csv_results/standard_depth_results.csv')
    # parser.add_argument('--out_dir', type=str, default='vis')
    # parser.add_argument('--remove_outliers', action='store_true', default=False)
    # args = parser.parse_args()

    out_dir = 'vis'

    generate_legend_markers()

    depth_csv = 'csv_results/standard_depth_results.csv'
    depth_df = pd.read_csv(depth_csv)

    for matches in ['splg', 'loma']:
        pose_csv = f'csv_results/standard_{matches}_slim_pose_results.csv'
        pose_df = pd.read_csv(pose_csv)


    # metrics = ['A.Rel_si', 'd1_si', 'A.Rel_ssi', 'd1_ssi']
    # metrics = ['d1_si', 'd1_ssi']
        dataset = 'mean'

        plot_scatter_pose_depth(pose_df, depth_df, dataset, 'd1_si',
                                remove_outliers=False,
                                out_dir=os.path.join(out_dir, 'sm'), iters=1000, matches=matches,
                                solvers=['calib', 'calib_ro', 'sf', 'sf_ro'])

        plot_scatter_pose_depth(pose_df, depth_df, dataset, 'd1_ssi',
                                remove_outliers=False,
                                out_dir=os.path.join(out_dir, 'sm'), iters=1000, matches=matches,
                                solvers=['calib_shift', 'calib_shift_ro', 'sf_shift', 'sf_shift_ro'])

        if matches == 'loma':
            plot_scatter_pose_depth(pose_df, depth_df, dataset, 'd1_si',
                                    remove_outliers=True, out_dir=out_dir,
                                    iters=1000, matches=matches, solvers=['calib', 'calib_ro'],
                                    figsize=(0.8*6, 0.8*4))

            for dataset in ['eth3d', 'lamar', 'sintel','scannetpp']:
                plot_scatter_pose_depth(pose_df, depth_df, dataset,
                                        'd1_si', remove_outliers=False, out_dir=os.path.join(out_dir, 'per_scene'),
                                        iters=1000, matches=matches, solvers=['calib', 'calib_ro'],
                                        figsize=(0.8*6, 0.8*4))


if __name__ == '__main__':
    main()