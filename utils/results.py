import copy
import json
import os

import h5py
import numpy as np
from gitdb.util import basename
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.cluster.hierarchy import single
from matplotlib.lines import Line2D

from utils.system_info import save_metadata


def get_summary_metrics(experiments, results):
    metrics = {}
    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]
        if not exp_results:
            continue

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        f_errs = np.array([r['f_err'] for r in exp_results])

        p_errs[np.isnan(p_errs)] = 180
        f_errs[np.isnan(f_errs)] = 1.0

        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 11)])
        f_res = np.array([np.sum(f_errs < t/100) / len(f_errs) for t in range(1, 11)])

        times = np.array([x['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        metrics[exp] = {
            'median_pose_err': np.median(p_errs),
            'median_f_err': np.median(f_errs),
            'pose_mAA_10': 100 * np.mean(p_res),
            'pose_mAA_5': 100 * np.mean(p_res[:5]),
            'pose_mAA_3': 100 * np.mean(p_res[:3]),
            'f_mAA_10': 100 * np.mean(f_res),
            'f_mAA_5': 100 * np.mean(f_res[:5]),
            'f_mAA_3': 100 * np.mean(f_res[:3]),
            'mean_runtime': np.mean(times) / 1e6,
            'mean_inliers': np.mean(inliers)
        }
    return metrics


def print_results_focal(metrics):
    tab = PrettyTable(['solver', 'median pose err', 'median f err',
                       'pose mAA(10)', 'f mAA(0.1)', 'mean mde runtime', 'mean pose runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp_name, m in metrics.items():
        tab.add_row([exp_name, m['median_pose_err'], m['median_f_err'],
                     m['pose_mAA_10'], m['f_mAA_10'], m['mean_mde_runtime'], m['mean_runtime'], m['mean_inliers']])
    print(tab)


def print_results_all(args, all_metrics=None):
    if all_metrics is None:
        all_metrics = merge_summary_results(args)

    tab = PrettyTable(['MDE', 'solver', 'median pose err', 'median f err',
                       'pose mAA(10)', 'f mAA(0.1)', 'mean runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for mde_name, metrics in all_metrics.items():
        for exp_name, m in metrics.items():
            tab.add_row([mde_name, exp_name, m['median_pose_err'], m['median_f_err'],
                         m['pose_mAA_10'], m['f_mAA_10'], m['mean_mde_runtime'], m['mean_inliers']])
    print(tab)

def draw_cumplots(experiments, results):
    plt.figure()
    plt.xlabel('Pose error')
    plt.ylabel('Portion of samples')

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]
        exp_name = exp
        label = f'{exp_name}'

        R_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 180)])
        plt.plot(np.arange(1, 180), R_res, label = label)

    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('k error')
    plt.ylabel('Portion of samples')


def get_results_dir(args, type='summary'):
    return os.path.join(args.data_path, f'{type}_results',
                        f'{args.name}_{args.matches}_{args.sampson_threshold}t_{args.reprojection_threshold}r')


def save_summary_results(experiments, full_results, mde_runtimes, args):
    results_dir = get_results_dir(args)
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f'{args.depth}.json')

    metrics = get_summary_metrics(experiments, full_results)
    for exp in experiments:
        metrics[exp]['mean_mde_runtime'] = np.mean(mde_runtimes)

    print_results_focal(metrics)

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_full_results(args, full_results):
    # Cache groups to avoid repeated lookups and string formatting

    h5_path = get_full_results_h5_path(args)

    with h5py.File(h5_path, 'w') as f_results:
        save_metadata(f_results)
        group_cache = {}
        for result in full_results:
            group_name = f"{result['image_name_1']}-{result['image_name_2']}"
            if group_name not in group_cache:
                group_cache[group_name] = f_results.require_group(group_name)
            group = group_cache[group_name]

            exp_group = group.create_group(result['experiment'])
            for key, value in result.items():
                if key in ('experiment', 'image_name_1', 'image_name_2'):
                    continue
                if isinstance(value, dict):
                    info_group = exp_group.create_group(key)
                    for k, v in value.items():
                        info_group.create_dataset(k, data=v)
                else:
                    exp_group.create_dataset(key, data=value)


def get_full_results_h5_path(args):
    results_dir = get_results_dir(args, 'full')
    os.makedirs(results_dir, exist_ok=True)
    h5_path = os.path.join(results_dir, f'{args.depth}.h5')
    return h5_path


def merge_summary_results(args):
    """Read all per-depth JSONs and merge into a single unified dict."""
    results_dir = get_results_dir(args)
    unified = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json') or fname == 'all.json':
            continue
        depth_name = fname[:-5]
        with open(os.path.join(results_dir, fname), 'r') as f:
            unified[depth_name] = json.load(f)
    return unified


def merge_summary_depth_results(args):
    """Read all per-depth JSONs and merge into a single unified dict."""
    results_dir = os.path.join(args.data_path, f'depth_results')
    unified = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json') or fname == 'all.json':
            continue
        depth_name = fname[:-5]
        with open(os.path.join(results_dir, fname), 'r') as f:
            unified[depth_name] = json.load(f)
    return unified


def load_full_results(f_results):
    full_results = []
    for group_name in f_results.keys():
        group = f_results[group_name]
        for exp_name in group.keys():
            exp_group = group[exp_name]
            res = {'experiment': exp_name}
            for key in exp_group.keys():
                if isinstance(exp_group[key], h5py.Group):
                    res[key] = {k: np.array(v) for k, v in exp_group[key].items()}
                else:
                    res[key] = np.array(exp_group[key])
            full_results.append(res)


def get_basename(args, depth: str) -> str:
    return (f'{args.name}_{args.matches}_{depth}'
            f'_{args.sampson_threshold}t_{args.reprojection_threshold}r')


def get_best_calib_result(all_metrics, variants):
    cal_solvers = ['calib', 'calib_shift']

    results = []
    for variant in variants:
        for k, v in all_metrics[variant].items():
            v['experiment'] = k
            v['variant'] = variant
        valid_variant_results = [all_metrics[variant][solver] for solver in cal_solvers]
        results.extend(valid_variant_results)
    mAA_results = [x['pose_mAA_10'] for x in results]
    return results[np.argmax(mAA_results)]

def get_best_uncal_result(all_metrics, variants):
    uncal_solvers = ['mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift']

    results = []
    for variant in variants:
        if 'Calib' in variant:
            continue
        for k, v in all_metrics[variant].items():
            v['experiment'] = k
            v['variant'] = variant
        valid_variant_results = [all_metrics[variant][solver] for solver in all_metrics[variant].keys() if solver in uncal_solvers]
        results.extend(valid_variant_results)
    mAA_results = [x['pose_mAA_10'] for x in results]
    return results[np.argmax(mAA_results)]


def print_best_only(all_metrics):
    mde_names = [x for x in all_metrics.keys()]

    variant_dict = {}

    for mde_name in mde_names:
        base_name = mde_name.split('-')[0].split('Calib')[0]
        if base_name in variant_dict:
            variant_dict[base_name].append(mde_name)
        else:
            variant_dict[base_name] = [mde_name]

    best_calib_results = {}
    best_uncal_results = {}
    for base_mde, variants in variant_dict.items():
        best_result = get_best_calib_result(all_metrics, variants)
        best_calib_results[best_result['variant']] ={best_result['experiment']: best_result}
        best_result = get_best_uncal_result(all_metrics, variants)
        best_uncal_results[best_result['variant']] ={best_result['experiment']: best_result}

    best_calib_results['gt'] = {'baseline_calib': all_metrics['gt']['baseline_calib']}

    if 'baseline_sf' in all_metrics['gt']:
        best_uncal_results['gt'] = {'baseline_sf': all_metrics['gt']['baseline_sf']}
    elif 'baseline_vf' in all_metrics['gt']:
        best_uncal_results['gt'] = {'baseline_vf': all_metrics['gt']['baseline_vf']}


    print("**** Best Calib Results ****")
    print_results_all(None, best_calib_results)
    print("**** Best Uncal Results ****")
    print_results_all(None, best_uncal_results)

    return best_calib_results, best_uncal_results


def get_mde_list(name, data_path):
    mde_list = [x.split('_depth_')[1].split('.h5')[0] for x in os.listdir(data_path)
                if x.startswith(f'{name}_depth_') and x.endswith('.h5')]

    return mde_list


def get_n_colors(n):
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in range(n)]
    return [plt.cm.hsv(i / n) for i in range(n)]


def plot_scatter_pose_depth(all_metrics, depth_metrics, name='default', remove_outliers=False):
    mde_list = list(depth_metrics.keys())

    if remove_outliers:
        mde_list = [x for x in mde_list if 'Infini' not in x and 'AnythingV2' not in x]
    if remove_outliers and name == 'lamar':
        mde_list = [x for x in mde_list if 'DepthPro' not in x]

    base_names = list(set([x.split('-')[0].split('Calib')[0] for x in mde_list]))

    single_metric = depth_metrics[mde_list[0]]
    depth_evals = [(k, x) for k, v in single_metric.items() for x in v.keys()]

    colors = get_n_colors(len(base_names))
    color_dict = {base_name: colors[i] for i, base_name in enumerate(base_names)}

    n = len(depth_evals)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    for solver in ['calib', 'calib_shift', 'mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift']:

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        fig.suptitle(f'Dataset: {name}, Solver: {solver}')
        axes = np.array(axes).flatten()

        for idx, (type, metric) in enumerate(depth_evals):
            ax = axes[idx]
            for mde in mde_list:
                base_name = mde.split('-')[0].split('Calib')[0]
                if 'Calib' in mde and ('vf' in solver or 'sf' in solver or 'mdecalib' in solver):
                    continue
                depth_val = depth_metrics[mde][type][metric]
                try:
                    pose_val = all_metrics[mde][solver]['pose_mAA_10']
                except:
                    continue
                marker = 'o' if 'Calib' in mde else '*'
                ax.plot(depth_val, pose_val, color=color_dict[base_name], marker=marker, linestyle='None')
            ax.set_xlabel(f"Depth {type} - {metric}")
            ax.set_ylabel("Pose mAA(10)")

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        color_handles = [Line2D([0], [0], color=c, marker='s', linestyle='None', label=name)
                         for name, c in color_dict.items()]
        marker_handles = [Line2D([0], [0], color='gray', marker='*', linestyle='None', label='Normal'),
                          Line2D([0], [0], color='gray', marker='o', linestyle='None', label='Calib')]

        fig.legend(handles=color_handles + marker_handles, loc='center right')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(f'vis/scatter_pose_depth_{name}_{solver}{"_no_outliers" if remove_outliers else "_all"}.png')
        # plt.show()


def plot_scatter_pose_depth_best(best_metrics, depth_metrics, version='calib', name='default', remove_outliers=False):
    mde_list = list(best_metrics.keys())

    mde_list = [x for x in mde_list if 'gt' != x]

    if remove_outliers:
        mde_list = [x for x in mde_list if 'Infini' not in x and 'AnythingV2' not in x]
    if remove_outliers and name == 'lamar':
        mde_list = [x for x in mde_list if 'DepthPro' not in x]

    single_metric = depth_metrics[mde_list[0]]
    depth_evals = [(k, x) for k, v in single_metric.items() for x in v.keys() if 'metric' not in k]

    base_names = list(set([x.split('-')[0].split('Calib')[0] for x in mde_list]))

    colors = get_n_colors(len(base_names))
    color_dict = {base_name: colors[i] for i, base_name in enumerate(base_names)}

    n = len(depth_evals)
    ncols = 2
    nrows = (n + ncols - 1) // ncols


    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    fig.suptitle(f'Dataset: {name}, Case: {version}')
    axes = np.array(axes).flatten()

    for idx, (type, metric) in enumerate(depth_evals):
        ax = axes[idx]
        for mde in mde_list:
            base_name = mde.split('-')[0].split('Calib')[0]
            depth_val = depth_metrics[mde][type][metric]
            try:
                pose_val = list(best_metrics[mde].items())[0][1]['pose_mAA_10']
            except:
                continue
            marker = 'o' if 'Calib' in mde else '*'
            ax.plot(depth_val, pose_val, color=color_dict[base_name], marker=marker, linestyle='None')
        ax.set_xlabel(f"Depth {type} - {metric}")
        ax.set_ylabel("Pose mAA(10)")

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        color_handles = [Line2D([0], [0], color=c, marker='s', linestyle='None', label=name)
                         for name, c in color_dict.items()]
        marker_handles = [Line2D([0], [0], color='gray', marker='*', linestyle='None', label='Normal'),
                          Line2D([0], [0], color='gray', marker='o', linestyle='None', label='Calib')]

        fig.legend(handles=color_handles + marker_handles, loc='center right')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(f'vis/scatter_pose_depth_{name}_best_{version}{"_no_outliers" if remove_outliers else "_all"}.png')
        # plt.show()


def parse_args():
    global args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('name')
    parser.add_argument('matches')
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('-st', '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-rt', '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to write unified JSON (default: <results_dir>/all.json)')
    parser.add_argument('-ed', '--eval_depth', action='store_true', default=False)
    args = parser.parse_args()
    return args

def main(args):
    all_metrics = merge_summary_results(args)

    output_path = args.output or os.path.join(get_results_dir(args), 'all.json')
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Unified results written to {output_path}\n")

    print_results_all(args, all_metrics)
    best_calib_results, best_uncal_results = print_best_only(all_metrics)
    if args.eval_depth:
        depth_metrics = merge_summary_depth_results(args)
        plot_scatter_pose_depth(all_metrics, depth_metrics, args.name)
        plot_scatter_pose_depth(all_metrics, depth_metrics, args.name, remove_outliers=True)
        plot_scatter_pose_depth_best(best_calib_results, depth_metrics, version='calib', name=args.name)
        plot_scatter_pose_depth_best(best_calib_results, depth_metrics, version='calib', name=args.name,
                                     remove_outliers=True)
        plot_scatter_pose_depth_best(best_uncal_results, depth_metrics, version='uncal', name=args.name)
        plot_scatter_pose_depth_best(best_uncal_results, depth_metrics, version='uncal', name=args.name,
                                     remove_outliers=True)


if __name__ == '__main__':
    args = parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            dataset_config = json.load(f)

        for name, config in dataset_config.items():
            single_args = copy.copy(args)
            single_args.name = name
            single_args.data_path = config["work_path"]
            main(single_args)
