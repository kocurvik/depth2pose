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
                       'pose mAA(10)', 'f mAA(0.1)', 'mean runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp_name, m in metrics.items():
        tab.add_row([exp_name, m['median_pose_err'], m['median_f_err'],
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