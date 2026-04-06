import json
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


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


def print_results_all(args):
    json_path = os.path.join(args.data_path,'summary_results',
                             f'{args.name}_{args.matches}_{args.sampson_threshold}t_{args.reprojection_threshold}r.json')
    with open(json_path, 'r') as f:
        all_metrics = json.load(f)

    tab = PrettyTable(['MDE', 'solver', 'median pose err', 'median f err',
                       'pose mAA(10)', 'f mAA(0.1)', 'mean runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for mde_name, metrics in all_metrics.items():
        for exp_name, m in metrics.items():
            tab.add_row([mde_name, exp_name, m['median_pose_err'], m['median_f_err'],
                         m['pose_mAA_10'], m['f_mAA_10'], m['mean_runtime'], m['mean_inliers']])
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


def save_summary_results(experiments, full_results, mde_runtimes, args):
    json_path = os.path.join(args.data_path, 'summary_results',
                             f'{args.name}_{args.matches}_{args.sampson_threshold}t_{args.reprojection_threshold}r.json')
    metrics = get_summary_metrics(experiments, full_results)
    for exp in experiments:
        metrics[exp]['mean_mde_runtime'] = np.mean(mde_runtimes)
    print_results_focal(metrics)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            summary_dict = json.load(f)
    else:
        summary_dict = {}
    if args.depth in summary_dict.keys():
        summary_dict[args.depth].update(metrics)
    else:
        summary_dict[args.depth] = metrics
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)


def save_full_results(f_results, full_results):
    for result in full_results:
        # write into f_results the result in group by image_name_1_image_name_2
        group_name = f"{result['image_name_1']}-{result['image_name_2']}"
        if group_name not in f_results:
            group = f_results.create_group(group_name)
        else:
            group = f_results[group_name]

        exp_group = group.create_group(result['experiment'])
        for key, value in result.items():
            if key in ['experiment', 'image_name_1', 'image_name_2']:
                continue
            if isinstance(value, dict):
                # Handle nested info dict
                info_group = exp_group.create_group(key)
                for k, v in value.items():
                    info_group.create_dataset(k, data=v)
            else:
                exp_group.create_dataset(key, data=value)


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
