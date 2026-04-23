import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from matplotlib.lines import Line2D


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(errors, thresholds):
    # code from https://github.com/cvg/GeoCalib/blob/982f625d822fbb2b50c65d123fd1362a5c01ec2c/siclib/utils/tools.py#L157
    errors, recall = compute_recall(errors)

    recall = np.r_[0, recall]
    errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t
        aucs.append(auc)
    return aucs


def get_summary_metrics(experiments, results, iters_list = (10, 100, 500, 1000)):
    metrics = {}
    for iters in iters_list:
        iters_results = [x for x in results if x['iterations'] == iters]
        if not iters_results:
            continue
        metrics[iters] = {}
        for exp in experiments:
            exp_results = [x for x in iters_results if x['experiment'] == exp]

            p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
            f_errs = np.array([r['f_err'] for r in exp_results])

            p_errs[np.isnan(p_errs)] = 180
            f_errs[np.isnan(f_errs)] = 1.0

            p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 11)])
            f_res = np.array([np.sum(f_errs < t/100) / len(f_errs) for t in range(1, 11)])

            times = np.array([x['runtime'] for x in exp_results])
            inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

            pose_mAA_10, pose_mAA_5, pose_mAA_3 = compute_auc(p_errs, [10, 5, 3])
            f_mAA_10, f_mAA_5, f_mAA_3 = compute_auc(f_errs, [0.1, 0.05, 0.03])

            metrics[iters][exp] = {
                'median_pose_err': np.median(p_errs),
                'median_f_err': np.median(f_errs),
                'pose_mAA_10_approx': 100 * np.mean(p_res),
                'pose_mAA_5_approx': 100 * np.mean(p_res[:5]),
                'pose_mAA_3_approx': 100 * np.mean(p_res[:3]),
                'f_mAA_10_approx': 100 * np.mean(f_res),
                'f_mAA_5_approx': 100 * np.mean(f_res[:5]),
                'f_mAA_3_approx': 100 * np.mean(f_res[:3]),
                'pose_mAA_10': 100 * pose_mAA_10,
                'pose_mAA_5': 100 * pose_mAA_5,
                'pose_mAA_3': 100 * pose_mAA_3,
                'f_mAA_10': 100 * f_mAA_10,
                'f_mAA_5_': 100 * f_mAA_5,
                'f_mAA_3_': 100 * f_mAA_3,
                'mean_runtime': np.mean(times) / 1e6,
                'mean_inliers': np.mean(inliers)
            }
    return metrics


def print_results_focal(metrics):
    tab = PrettyTable(['solver', 'iters', 'median pose err', 'median f err',
                       'pose mAA(10)', 'f mAA(0.1)', 'mean mde runtime', 'mean pose runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for iters, iter_metrics in metrics.items():
        for exp_name, m in iter_metrics.items():
            tab.add_row([exp_name, iters, m['median_pose_err'], m['median_f_err'],
                         m['pose_mAA_10'], m['f_mAA_10'],
                         m['mean_mde_runtime'], m['mean_runtime'],
                         m['mean_inliers']])
    print(tab)


def print_results_all(args, all_metrics=None):
    if all_metrics is None:
        all_metrics = merge_summary_results(args)

    tab = PrettyTable(['MDE', 'solver', 'iters', 'median pose err', 'median f err',
                       'pose mAA(10)', 'f mAA(0.1)', 'mean runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for mde_name, metrics in all_metrics.items():
        for iters, iter_metrics in metrics.items():
            for exp_name, m in iter_metrics.items():
                tab.add_row([mde_name, exp_name, iters, m['median_pose_err'], m['median_f_err'],
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
                        f'{args.dataset_name}_{args.matches}_{args.sampson_threshold}t_{args.reprojection_threshold}r')


def save_summary_results(experiments, full_results, mde_runtimes, args):
    results_dir = get_results_dir(args)
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f'{args.depth}.json')

    metrics = get_summary_metrics(experiments, full_results)
    for iter_metrics in metrics.values():
        for m in iter_metrics.values():
            m['mean_mde_runtime'] = np.mean(mde_runtimes)

    print_results_focal(metrics)

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)


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


def get_basename(args, depth: str) -> str:
    return (f'{args.dataset_name}_{args.matches}_{depth}'
            f'_{args.sampson_threshold}t_{args.reprojection_threshold}r')


def get_best_result(all_metrics, variants, uncal=False, ro=False):
    if uncal:
        solvers = ['mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift']
    else:
        solvers = ['calib', 'calib_shift']

    if ro:
        solvers = [f'{x}_ro' for x in solvers]

    results = []
    for variant in variants:
        for k, v in all_metrics[variant].items():
            if 'Calib' in variant and uncal:
                continue
            v['experiment'] = k
            v['variant'] = variant
        valid_variant_results = [all_metrics[variant][solver] for solver in solvers]
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
        best_result = get_best_result(all_metrics, variants)
        best_calib_results[best_result['variant']] = {best_result['experiment']: best_result}
        best_result = get_best_result(all_metrics, variants, uncal=True)
        best_uncal_results[best_result['variant']] = {best_result['experiment']: best_result}

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

    # colors = get_n_colors(len(base_names))
    # color_dict = {base_name: colors[i] for i, base_name in enumerate(base_names)}
    color_dict = get_mde_basename_color_dict()

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
                         for name, c in color_dict.items() if name in base_names]
        marker_handles = [Line2D([0], [0], color='gray', marker='*', linestyle='None', label='Normal'),
                          Line2D([0], [0], color='gray', marker='o', linestyle='None', label='Calib')]

        fig.legend(handles=color_handles + marker_handles, loc='center right')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(f'vis/scatter_pose_depth_{name}_best_{version}{"_no_outliers" if remove_outliers else "_all"}.png')
        # plt.show()


def flatten_depth_metrics(all_metrics):
    rows = []

    for mde, mde_metrics in all_metrics.items():
        row = {'mde': mde}
        for eval_type, metrics in mde_metrics.items():
            row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_pose_metrics(all_metrics):
    rows = []
    for mde, mde_metrics in all_metrics.items():
        for iters, iter_metrics in mde_metrics.items():
            for solver, metrics in iter_metrics.items():
                row = {'mde': mde, 'iters': int(iters), 'solver': solver}
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)