import argparse
from multiprocessing import Process, Queue, Pool
import time
import os
import signal
from time import perf_counter_ns

# from time import perf_counter

import h5py
import numpy as np
import poselib
from tqdm import tqdm

from utils.geometry import R_err_fun, t_err_fun, get_kp_depth
from utils.multiprocessing import NoDaemonProcessPool
from utils.results import save_summary_results, print_results_all, save_full_results, load_full_results
from utils.system_info import save_metadata

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
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('data_path')
    parser.add_argument('name')
    parser.add_argument('matches')


    return parser.parse_args()

def get_result_dict(info, monodepth_pose, R_gt, t_gt, f1_gt, f2_gt, camera1=None, camera2=None):
    out = {}

    pose_est = monodepth_pose.geometry.pose
    R_est, t_est = pose_est.R, pose_est.t

    out['R'] = R_est
    out['R_gt'] = R_gt
    out['t'] = t_est
    out['t_gt'] = t_gt
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

    info['inliers'] = np.array(info['inliers'])
    out['info'] = info

    return out


def get_exception_result_dict(x):
    experiment = x[1]
    return {'experiment': experiment, 'R_err': 180.0, 't_err': 180.0, 'f_err': 1e6,
            'f1_err': 1e6, 'f2_err': 1e6, 'info': {'inliers': []}}


def eval_experiment(x):
    experiment, kp1, kp2, d1, d2, K1_mde, K2_mde, R_gt, t_gt, K1_gt, K2_gt, img_name_1, img_name_2, t, r = x

    f1_gt = (K1_gt[0, 0] + K1_gt[1, 1]) / 2
    f2_gt = (K2_gt[0, 0] + K2_gt[1, 1]) / 2

    shift = 'shift' in experiment

    bundle_dict = {'max_iterations': 100, 'verbose': False}
    ransac_dict = {'max_iterations': 1000, 'min_iterations': 1000, 'progressive_sampling': False}

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
        bundle_dict['loss_type'] = 'TRUNCATED_CAUCHY'
        monodepth_dict = {'max_errors': [r, t], 'estimate_shift': shift, 'ransac': ransac_dict}
        start_time = perf_counter_ns()
        monodepth_pose, info = poselib.estimate_monodepth_relative_pose(kp1, kp2, d1, d2,
                                                                        camera1, camera2,
                                                                        monodepth_dict)
        runtime = perf_counter_ns() - start_time

        monodepth_pair = poselib.MonoDepthImagePair(monodepth_pose, camera1, camera2)

    if 'baseline' == experiment:
        relpose_dict = {'max_error': t, 'ransac': ransac_dict, 'bundle': bundle_dict}
        start_time = perf_counter_ns()
        pose, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, relpose_dict)
        runtime = perf_counter_ns() - start_time
        monodepth_pair = poselib.MonoDepthImagePair(poselib.MonoDepthTwoViewGeometry(pose), camera1, camera2)

    result_dict = get_result_dict(info, monodepth_pair, R_gt, t_gt, f1_gt, f2_gt, camera1=camera1, camera2=camera2)
    result_dict['experiment'] = experiment
    result_dict['runtime'] = runtime
    result_dict['image_name_1'] = img_name_1
    result_dict['image_name_2'] = img_name_2

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


def get_gt_depth(kp1, kp2, R_gt, t_gt, K1_gt, K2_gt):
    # implement a function that triangulates the points to obtain depth for each of the two views

    kp1_norm = np.linalg.inv(K1_gt) @ np.hstack((kp1, np.ones((kp1.shape[0], 1)))).T
    kp2_norm = np.linalg.inv(K2_gt) @ np.hstack((kp2, np.ones((kp2.shape[0], 1)))).T

    # Triangulate to get 3D points in camera 1 coordinate system
    # Using the standard triangulation formula for relative pose
    # x2 ~ R*x1 + t  =>  lambda2 * x2 = lambda1 * R * x1 + t
    # [R*x1  -x2] [lambda1; lambda2] = -t

    d1 = np.zeros(kp1.shape[0])
    d2 = np.zeros(kp1.shape[0])

    for i in range(kp1.shape[0]):
        A = np.zeros((3, 2))
        A[:, 0] = R_gt @ kp1_norm[:, i]
        A[:, 1] = -kp2_norm[:, i]
        b = -t_gt.flatten()

        # Solve least squares for depths (lambdas)
        depths, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        d1[i] = depths[0]
        d2[i] = depths[1]

    return d1, d2


def eval_single_mde(args):
    experiments = ['calib']

    if args.include_mde_K:
        experiments.append('mdecalib')

    if args.include_shared_focal:
        if 'Calib' in args.depth:
            print("Shared focal solver requested, but MDE used GT calibration. Skipping.")
        else:
            experiments.append('sf')

    if args.include_varying_focal:
        if 'Calib' in args.depth:
            print("Varying focal solver requested, but MDE used GT calibration. Skipping.")
        else:
            experiments.append('vf')

    if args.include_shift_solvers:
        experiments.extend([f'{x}_shift' for x in experiments])

    if args.include_baseline_solver:
        experiments.append('baseline')

    print(f"Running: {experiments}")

    basename = f'{args.name}_{args.matches}_{args.depth}_{args.sampson_threshold}t_{args.reprojection_threshold}r'

    os.makedirs(os.path.join(args.data_path, 'full_results'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'summary_results'), exist_ok=True)

    h5_path = os.path.join(args.data_path, f'full_results/{basename}.h5')

    if args.load:
        print("Loading: ", h5_path)        
        with h5py.File(h5_path, 'r') as f_results:
            load_full_results(f_results)
    else:
        name_path = os.path.join(args.data_path, args.name)

        image_pair_list_path = f'{name_path}_image_pairs.txt'
        with open(image_pair_list_path, 'r') as f:
            pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

        image_list_path = f'{name_path}_image_list.txt'
        with open(image_list_path, 'r') as f:
            image_list = [x.strip() for x in f.readlines()]
        
        with h5py.File(f'{name_path}.h5') as f_images_h5:        
            f_images = {}
            for image_name in image_list:
                f_images[f'{image_name}_K'] = np.array(f_images_h5[f'{image_name}_K'])
                f_images[f'{image_name}_R'] = np.array(f_images_h5[f'{image_name}_R'])
                f_images[f'{image_name}_T'] = np.array(f_images_h5[f'{image_name}_T'])
            
        with h5py.File(f'{name_path}_{args.matches}.h5') as f_matches_h5:
            f_matches = {}
            for image_name_1, image_name_2 in pair_list:
                f_matches[f"{image_name_1}-{image_name_2}"] = np.array(f_matches_h5[f"{image_name_1}-{image_name_2}"])

        
        if args.depth != 'gt':            
            with h5py.File(f'{name_path}_depth_{args.depth}.h5', 'r') as f_depth_h5:
                if 'completed' not in f_depth_h5:
                    raise ValueError(f'{name_path}_depth_{args.depth}.h5 does not have the completed tag. Aborting.')

                mde_runtimes = [f_depth_h5[f'{x}_runtime'][()] / 1e6 for x in image_list]
                
                f_depth = {}
                for image_name in image_list:
                    f_depth[f'{image_name}_depth'] = np.array(f_depth_h5[f'{image_name}_depth'])
                    if f'{image_name}_K' in f_depth_h5:
                        f_depth[f'{image_name}_K'] = np.array(f_depth_h5[f'{image_name}_K'])
            
        else:
            f_depth = None
            mde_runtimes = [0 for x in image_list]


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

                if args.depth != 'gt':
                    depth_map1 = f_depth[f'{img_name_1}_depth']
                    depth_map2 = f_depth[f'{img_name_2}_depth']

                    d1 = get_kp_depth(kp1, depth_map1, interpolation='nearest')
                    d2 = get_kp_depth(kp2, depth_map2, interpolation='nearest')

                    l = np.logical_and(np.isfinite(d1), np.isfinite(d2))
                    kp1 = kp1[l]
                    kp2 = kp2[l]
                    d1 = d1[l]
                    d2 = d2[l]
                else:
                    d1, d2 = get_gt_depth(kp1, kp2, R_gt, t_gt, K1_gt, K2_gt)

                try:
                    mde_K1 = np.array(f_depth[f'{img_name_1}_K'])
                    mde_K2 = np.array(f_depth[f'{img_name_2}_K'])
                except Exception:
                    global MDE_K_WARNING_SHOWN

                    if not MDE_K_WARNING_SHOWN:
                        print("Warning: MDE K matrices not found in depth file. Using None.")
                        MDE_K_WARNING_SHOWN = True
                        print("Removing variants using MDE K from experiments.")
                        experiments[:] = [e for e in experiments if 'mdecalib' not in e]
                        print(f'Experiments {experiments}')
                    mde_K1, mde_K2 = None, None

                for experiment in experiments:
                    yield (experiment, np.copy(kp1), np.copy(kp2), np.copy(d1), np.copy(d2),
                           mde_K1, mde_K2, R_gt, t_gt, K1_gt, K2_gt, img_name_1, img_name_2,
                           args.sampson_threshold, args.reprojection_threshold)

        total_length = len(experiments) * len(pair_list)

        print(f"Total runs: {total_length} for {len(pair_list)} samples")

        if args.num_workers == 1:
            full_results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            if args.timeout_pool:
                pool = NoDaemonProcessPool(args.num_workers)
                full_results = [x for x in pool.imap(run_with_timeout, tqdm(gen_data(), total=total_length))]
            else:
                pool = Pool(args.num_workers)
                full_results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        with h5py.File(h5_path, 'w') as f_results:
            save_metadata(f_results)
            save_full_results(f_results, full_results)        

        save_summary_results(experiments, full_results, mde_runtimes, args)


if __name__ == '__main__':
    args = parse_args()
    if args.depth is None:
        mde_list = [x.split('_depth_')[1].split('.h5')[0] for x in os.listdir(args.data_path)
                    if x.startswith(f'{args.name}_depth_') and x.endswith('.h5')]

        for depth_name in mde_list:
            print(f"Checking if MDE {depth_name} results are available!")
            basename = f'{args.name}_{args.matches}_{args.depth}_{args.sampson_threshold}t_{args.reprojection_threshold}r'
            h5_path = os.path.join(args.data_path, f'full_results/{basename}.h5')
            if os.path.exists(h5_path) and not args.recalc:
                print(f"Results in {h5_path} available. Skipping")
                continue

            print(f"Running for MDE: {depth_name}")
            args.depth = depth_name
            try:
                eval_single_mde(args)
            except ValueError as e:
                print(e)
                print("Skipping...")


        print("Running for GT depth")
        args.depth = 'gt'
        args.include_baseline_solver = True
        eval_single_mde(args)

        print_results_all(args)
    else:
        eval_single_mde(args)