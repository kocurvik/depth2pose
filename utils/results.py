import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def print_results_focal(experiments, results):
    tab = PrettyTable(['solver', 'median pose err', 'median f err',
                       'pose mAA', 'f mAA', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        f_errs = np.array([r['f_err'] for r in exp_results])

        p_errs[np.isnan(p_errs)] = 180
        f_errs[np.isnan(f_errs)] = 1.0

        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 11)])
        f_res = np.array([np.sum(f_errs < t/100) / len(f_errs) for t in range(1, 11)])

        # times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        exp_name = exp


        tab.add_row([exp_name, np.median(p_errs), np.median(f_errs),
                     100 * np.mean(p_res), 100 * np.mean(f_res),
                     # np.mean(times),
                     np.mean(inliers)])
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