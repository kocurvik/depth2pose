import argparse
import json
from multiprocessing import Process, Queue, Pool
import time
import os
import signal
# from time import perf_counter

import h5py
import numpy as np
import poselib
from tqdm import tqdm

from utils.geometry import R_err_fun, t_err_fun, get_kp_depth
from utils.multiprocessing import NoDaemonProcessPool
from utils.results import print_results_focal, draw_cumplots

MDE_K_WARNING_SHOWN = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st',  '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-sw',  '--sampson_weight', type=float, default=1.0)
    parser.add_argument('-rt',  '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('-mk',  '--include_mde_K', action='store_true', default=False)
    parser.add_argument('-bs',  '--include_baseline_solver', action='store_true', default=False)
    parser.add_argument('-ss',  '--include_shift_solvers', action='store_true', default=False)
    parser.add_argument('-sf',  '--include_shared_focal', action='store_true', default=False)
    parser.add_argument('-vf',  '--include_varying_focal', action='store_true', default=False)
    parser.add_argument('--timeout_pool', action='store_true', default=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('data_path')
    parser.add_argument('name')
    parser.add_argument('matches')
    parser.add_argument('depths')

    return parser.parse_args()

def get_result_dict(info, monodepth_pose, R_gt, t_gt, f1_gt, f2_gt, camera1=None, camera2=None):
    out = {}

    pose_est = monodepth_pose.geometry.pose
    R_est, t_est = pose_est.R, pose_est.t

    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = t_est.tolist()
    out['t_gt'] = t_gt.tolist()
    out['f1_gt'] = f1_gt
    out['f2_gt'] = f2_gt

    if camera1 is None:
        camera1 = monodepth_pose.camera1
    if camera2 is None:
        camera2 = monodepth_pose.camera2
    out['f1'] = camera1.focal()
    out['f2'] = camera2.focal()
    out['R_err'] = R_err_fun(R_est, R_gt)
    out['t_err'] = t_err_fun(t_est, t_gt)

    out['f1_err'] = np.abs(out['f1'] - f1_gt) / f1_gt
    out['f2_err'] = np.abs(out['f2'] - f2_gt) / f2_gt
    out['f_err'] = np.sqrt(out['f1_err'] * out['f2_err'])

    info['inliers'] = []
    out['info'] = info

    return out


def get_exception_result_dict(x):
    experiment = x[1]
    return {'experiment': experiment, 'R_err': 180.0, 't_err': 180.0, 'f_err': 1e6,
            'f1_err': 1e6, 'f2_err': 1e6, 'info': {'inliers': []}}


def eval_experiment(x):
    experiment, kp1, kp2, d1, d2, K1_mde, K2_mde, R_gt, t_gt, K1_gt, K2_gt, t, r = x

    f1_gt = (K1_gt[0, 0] + K1_gt[1, 1]) / 2
    f2_gt = (K2_gt[0, 0] + K2_gt[1, 1]) / 2

    shift = 'shift' in experiment

    bundle_dict = {'max_iterations': 100, 'verbose': False, 'loss_type': 'TRUNCATED_CAUCHY'}
    ransac_dict = {'max_iterations': 10000, 'min_iterations': 10000, 'progressive_sampling': False}
    monodepth_dict = {'max_errors': [t, r],  'estimate_shift': shift, 'ransac': ransac_dict, 'bundle': bundle_dict}

    if 'mdecalib' in experiment:
        camera1 = {'model': 'PINHOLE', 'width': -1, 'height': -1,
                   'params': [K1_mde[0, 0], K1_mde[1, 1], K1_mde[0, 2], K1_mde[1, 2]]}
        camera2 = {'model': 'PINHOLE', 'width': -1, 'height': -1,
                   'params': [K2_mde[0, 0], K2_mde[1, 1], K2_mde[0, 2], K2_mde[1, 2]]}

    else:
        camera1 = {'model': 'PINHOLE', 'width': -1, 'height': -1,
                   'params': [K1_gt[0, 0], K1_gt[1, 1], K1_gt[0, 2], K1_gt[1, 2]]}
        camera2 = {'model': 'PINHOLE', 'width': -1, 'height': -1,
                   'params': [K2_gt[0, 0], K2_gt[1, 1], K2_gt[0, 2], K2_gt[1, 2]]}

    camera1 = poselib.Camera(camera1)
    camera2 = poselib.Camera(camera2)

    if 'calib' in experiment:
        monodepth_pose, info = poselib.estimate_monodepth_relative_pose(kp1, kp2, d1, d2,
                                                                        camera1, camera2,
                                                                        monodepth_dict)

        monodepth_pair = poselib.MonoDepthImagePair(monodepth_pose, camera1, camera2)

    if 'baseline' == experiment:
        relpose_dict = {'max_error': t, 'ransac': ransac_dict, 'bundle': bundle_dict}
        pose, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, relpose_dict)
        monodepth_pair = poselib.MonoDepthImagePair(poselib.MonoDepthTwoViewGeometry(pose), camera1, camera2)


    result_dict = get_result_dict(info, monodepth_pair, R_gt, t_gt, f1_gt, f2_gt, camera1=camera1, camera2=camera2)
    result_dict['experiment'] = experiment

    return result_dict


def eval_experiment_wrapper(x, result_queue):
    pid = os.getpid()

    try:
        result = eval_experiment(x)
        result_queue.put(result)
    except Exception as e:
        print(f"Process {pid}: Error in experiment: {e}")
        result_queue.put(get_exception_result_dict(x))


def run_with_timeout(x, timeout=20):
    result_queue = Queue()
    process = Process(target=eval_experiment_wrapper, args=(x, result_queue))
    process.start()
    process_pid = process.pid
    process.join(timeout)

    if process.is_alive():
        print(f"Process {process_pid} timed out after {timeout} seconds. Terminating...")
        process.terminate()
        time.sleep(0.1)
        if process.is_alive():
            print(f"Process {process_pid} didn't terminate. Sending SIGKILL...")
            try:
                os.kill(process.pid, signal.SIGKILL)
            except OSError:
                pass
        process.join(1)
        return get_exception_result_dict(x)

    if not result_queue.empty():
        return result_queue.get()
    else:
        return get_exception_result_dict(x)


def eval(args):
    experiments = ['calib']

    if args.include_mde_K:
        experiments.append('mdecalib')

    if args.include_shared_focal:
        experiments.append('sf')

    if args.include_varying_focal:
        experiments.append('vf')

    if args.include_shift_solvers:
        experiments.extend([f'{x}_shift' for x in experiments])

    if args.include_baseline_solver:
        experiments.append('baseline')

    print(experiments)


    basename = f'{args.name}_{args.matches}_{args.depths}_{args.sampson_threshold}t_{args.reprojection_threshold}r'

    json_string = f'{basename}.json'
    json_path = json_string

    if args.load:
        print("Loading: ", json_string)
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        name_path = os.path.join(args.data_path, args.name)

        # image_list_path = f'{name_path}_image_list.txt'
        # with open(image_list_path, 'r') as f:
        #     image_list = [x.strip() for x in f.readlines()]

        image_pair_list_path = f'{name_path}_image_pairs.txt'
        with open(image_pair_list_path, 'r') as f:
            pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

        f_images = h5py.File(f'{name_path}.h5')
        f_matches = h5py.File(f'{name_path}_{args.matches}.h5')
        f_depth = h5py.File(f'{name_path}_depth_{args.depths}.h5', 'r')


        if args.first is not None:
            pair_list = pair_list[:args.first]

        def gen_data():
            for img_name_1, img_name_2 in pair_list:
                K1_gt = np.array(f_images[f'{img_name_1}_K'])
                R1 = np.array(f_images[f'{img_name_1}_R'])
                T1 = np.array(f_images[f'{img_name_1}_T'])

                K2_gt = np.array(f_images[f'{img_name_2}_K'])
                R2 = np.array(f_images[f'{img_name_2}_R'])
                T2 = np.array(f_images[f'{img_name_2}_T'])

                R_gt = np.dot(R2, R1.T)
                t_gt = T2 - np.dot(R_gt, T1)

                kps = np.array(f_matches[f"{img_name_1}-{img_name_2}"])

                kp1 = kps[:, :2]
                kp2 = kps[:, 2:4]

                depth_map1 = np.array(f_depth[f'{img_name_1}_depth'])
                depth_map2 = np.array(f_depth[f'{img_name_2}_depth'])

                d1 = get_kp_depth(kp1, depth_map1, interpolation='linear')
                d2 = get_kp_depth(kp2, depth_map2, interpolation='linear')

                try:
                    mde_K1 = np.array(f_depth[f'{img_name_1}_K'])
                    mde_K2 = np.array(f_depth[f'{img_name_2}_K'])
                except Exception:
                    global MDE_K_WARNING_SHOWN
                    if not MDE_K_WARNING_SHOWN:
                        print("Warning: MDE K matrices not found in depth file. Using None.")
                        MDE_K_WARNING_SHOWN = True
                    mde_K1, mde_K2 = None, None

                for experiment in experiments:
                    yield (experiment, np.copy(kp1), np.copy(kp2), np.copy(d1), np.copy(d2),
                           mde_K1, mde_K2, R_gt, t_gt, K1_gt, K2_gt,
                           args.sampson_threshold, args.reprojection_threshold)

        total_length = len(experiments) * len(pair_list)

        print(f"Total runs: {total_length} for {len(pair_list)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            if args.timeout_pool:
                pool = NoDaemonProcessPool(args.num_workers)
                results = [x for x in pool.imap(run_with_timeout, tqdm(gen_data(), total=total_length))]
            else:
                pool = Pool(args.num_workers)
                results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        # os.makedirs('results', exist_ok=True)

        # if args.append:
        #     print(f"Appending from: {json_path}")
        #     try:
        #         with open(json_path, 'r') as f:
        #             prev_results = json.load(f)
        #     except Exception:
        #         print("Prev results not found!")
        #         prev_results = []
        #
        #     if args.overwrite:
        #         print("Overwriting old results")
        #         prev_results = [x for x in prev_results if not isinstance(x, tuple) and not isinstance(x, list)]
        #         prev_results = [x for x in prev_results if x['experiment'] not in experiments]
        #
        #     results.extend(prev_results)

        with open(json_path, 'w') as f:
            json.dump(results, f)

        print("Done")

    print_results_focal(experiments, results)
    draw_cumplots(experiments, results)

if __name__ == '__main__':
    args = parse_args()
    eval(args)