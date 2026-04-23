import argparse
import shutil
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

from utils.geometry import R_err_fun, t_err_fun, get_kp_depth, get_gt_inlier_mask
from utils.mp import NoDaemonProcessPool
from utils.results import save_summary_results, print_results_all, get_mde_list
from utils.storage import encode_result, save_full_results, get_full_results_h5_path, load_full_results

MDE_K_WARNING_SHOWN = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st',  '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-sw',  '--sampson_weight', type=float, default=1.0)
    parser.add_argument('-rt',  '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('-nmk', '--no_mde_K', action='store_true', default=False)
    parser.add_argument('-bs',  '--include_baseline_solver', action='store_true', default=False)
    parser.add_argument('-nss', '--no_shift_solvers', action='store_true', default=False)
    parser.add_argument('-sf',  '--include_shared_focal', action='store_true', default=False)
    parser.add_argument('-vf',  '--include_varying_focal', action='store_true', default=False)
    parser.add_argument('-nro', '--no_reproj_only_ransac', action='store_true', default=False)
    parser.add_argument('-dr',  '--direct_read', action='store_true', default=False)
    parser.add_argument('--timeout_pool', action='store_true', default=False)
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('--explicit_solvers', type=str, default=None)
    parser.add_argument('data_path')
    parser.add_argument('name')
    parser.add_argument('matches')


    return parser.parse_args()

def get_result_dict(info, monodepth_pose, R_gt, t_gt, f1_gt, f2_gt):
    out = {}

    pose_est = monodepth_pose.geometry.pose
    R_est, t_est = pose_est.R, pose_est.t

    out['R'] = R_est
    out['R_gt'] = R_gt
    out['t'] = t_est
    out['t_gt'] = t_gt
    out['f1_gt'] = f1_gt
    out['f2_gt'] = f2_gt

    camera1 = monodepth_pose.camera1
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
    (experiment, iters, kp1, kp2, d1, d2, K1_mde, K2_mde, pp_center_1, pp_center_2, R_gt, t_gt, cam1_gt, cam2_gt,
     img_name_1, img_name_2, gt_inlier_mask, t, r) = x

    f1_gt = (cam1_gt['params'][0] + cam1_gt['params'][1]) / 2
    f2_gt = (cam2_gt['params'][0] + cam2_gt['params'][1]) / 2
    # pp1 = np.array([cam1_gt['params'][2], cam1_gt['params'][3]])
    # pp2 = np.array([cam2_gt['params'][2], cam2_gt['params'][3]])

    shift = 'shift' in experiment

    bundle_dict = {'max_iterations': 100, 'verbose': False, 'loss_type': 'TRUNCATED_CAUCHY'}
    ransac_dict = {'max_iterations': iters, 'min_iterations': iters, 'progressive_sampling': False}

    if 'mdecalib' in experiment:
        camera1 = poselib.Camera({'model': 'PINHOLE', 'width': -1, 'height': -1,
                                  'params': [K1_mde[0, 0], K1_mde[1, 1], K1_mde[0, 2], K1_mde[1, 2]]})
        camera2 = poselib.Camera({'model': 'PINHOLE', 'width': -1, 'height': -1,
                                  'params': [K2_mde[0, 0], K2_mde[1, 1], K2_mde[0, 2], K2_mde[1, 2]]})
    else:
        camera1 = poselib.Camera(cam1_gt)
        camera2 = poselib.Camera(cam2_gt)

    monodepth_dict = {'max_errors': [r, t], 'estimate_shift': shift, 'ransac': ransac_dict}

    if '_ro' in experiment:
        monodepth_dict['weight_sampson'] = -1.0
        bundle_dict['loss_type'] = 'CAUCHY'
        kp1 = kp1[gt_inlier_mask]
        kp2 = kp2[gt_inlier_mask]
        d1 = d1[gt_inlier_mask]
        d2 = d2[gt_inlier_mask]

    if 'baseline_calib' == experiment:
        bundle_dict['loss_type'] = 'CAUCHY'
        relpose_dict = {'max_error': t, 'ransac': ransac_dict, 'bundle': bundle_dict}
        start_time = perf_counter_ns()
        pose, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, relpose_dict)
        runtime = perf_counter_ns() - start_time
        monodepth_pair = poselib.MonoDepthImagePair(poselib.MonoDepthTwoViewGeometry(pose), camera1, camera2)
    elif 'baseline_sf' == experiment:
        bundle_dict['loss_type'] = 'CAUCHY'
        relpose_dict = {'max_error': t, 'ransac': ransac_dict, 'bundle': bundle_dict}
        start_time = perf_counter_ns()
        image_pair, info = poselib.estimate_shared_focal_relative_pose(kp1, kp2, (pp_center_1 + pp_center_2) / 2,
                                                                       relpose_dict)
        runtime = perf_counter_ns() - start_time
        monodepth_pair = poselib.MonoDepthImagePair(poselib.MonoDepthTwoViewGeometry(image_pair.pose),
                                                    image_pair.camera1, image_pair.camera2)
    elif 'baseline_vf' == experiment:
        bundle_dict['loss_type'] = 'CAUCHY'
        relpose_dict = {'max_error': t, 'ransac': ransac_dict, 'bundle': bundle_dict}
        start_time = perf_counter_ns()
        image_pair, info = poselib.estimate_varying_focal_relative_pose(kp1, kp2, pp_center_1, pp_center_2,
                                                                        relpose_dict)
        runtime = perf_counter_ns() - start_time
        monodepth_pair = poselib.MonoDepthImagePair(poselib.MonoDepthTwoViewGeometry(image_pair.pose),
                                                    image_pair.camera1, image_pair.camera2)
    elif 'calib' in experiment:
        start_time = perf_counter_ns()
        monodepth_pose, info = poselib.estimate_monodepth_relative_pose(kp1, kp2, d1, d2,
                                                                        camera1, camera2,
                                                                        monodepth_dict)
        runtime = perf_counter_ns() - start_time

        monodepth_pair = poselib.MonoDepthImagePair(monodepth_pose, camera1, camera2)

    elif 'sf' in experiment:
        start_time = perf_counter_ns()
        monodepth_pair, info = poselib.estimate_monodepth_shared_focal_relative_pose(kp1 - pp_center_1,
                                                                                     kp2 - pp_center_2, d1, d2,
                                                                                     monodepth_dict)
        runtime = perf_counter_ns() - start_time

    elif 'vf' in experiment:
        start_time = perf_counter_ns()
        monodepth_pair, info = poselib.estimate_monodepth_varying_focal_relative_pose(kp1 - pp_center_1,
                                                                                      kp2 - pp_center_2, d1, d2,
                                                                                      monodepth_dict)
        runtime = perf_counter_ns() - start_time

    result_dict = get_result_dict(info, monodepth_pair, R_gt, t_gt, f1_gt, f2_gt)
    result_dict['experiment'] = experiment
    result_dict['runtime'] = runtime
    result_dict['image_name_1'] = img_name_1
    result_dict['image_name_2'] = img_name_2
    result_dict['iterations'] = iters
    result_dict['encoded'] = encode_result(result_dict)

    # test_dict = decode_result(result_dict['encoded'])
    # if runtime / 1e6 > 300:
    #     print(info)
    # print(f'For experimet: {experiment} runtime: {runtime / 1e6}')

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
    experiments = get_solvers(args)

    iters_list = [10, 100, 500, 1000]

    print(f"Running: {experiments}")

    basename = f'{args.name}_{args.matches}_{args.depth}_{args.sampson_threshold}t_{args.reprojection_threshold}r'

    os.makedirs(os.path.join(args.data_path, 'full_results'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'summary_results'), exist_ok=True)

    name_path = os.path.join(args.data_path, args.name)

    image_list_path = f'{name_path}_image_list.txt'
    with open(image_list_path, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]

    if args.load:
        full_results = load_full_results(args)

        if args.depth == 'gt':
            mde_runtimes = [0 for x in image_list]
        else:
            with h5py.File(f'{name_path}_depth_{args.depth}.h5', 'r') as f_depth_h5:
                if 'completed' not in f_depth_h5:
                    raise ValueError(f'{name_path}_depth_{args.depth}.h5 does not have the completed tag. Aborting.')

                mde_runtimes = [f_depth_h5[f'{x}_runtime'][()] / 1e6 for x in image_list]

        save_summary_results(experiments, full_results, mde_runtimes, args)
    else:
        image_pair_list_path = f'{name_path}_image_pairs.txt'
        with open(image_pair_list_path, 'r') as f:
            pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

        with h5py.File(f'{name_path}.h5') as f_images_h5:        
            f_images = {}
            for key in f_images_h5.keys():
                f_images[key] = np.array(f_images_h5[key])

            
        with h5py.File(f'{name_path}_{args.matches}.h5') as f_matches_h5:
            f_matches = {}
            for image_name_1, image_name_2 in pair_list:
                f_matches[f"{image_name_1}-{image_name_2}"] = np.array(f_matches_h5[f"{image_name_1}-{image_name_2}"])

        
        if args.depth != 'gt':
            if args.direct_read:
                try:
                    job_id = os.environ.get('SLURM_JOB_ID', 'local')
                    print(f'Copying {name_path}_depth_{args.depth}.h5 to /work/{job_id}/{args.name}_depth_{args.depth}.h5')
                    shutil.copy(f'{name_path}_depth_{args.depth}.h5', f'/work/{job_id}/{args.name}_depth_{args.depth}.h5')
                    f_depth = h5py.File(f'/work/{job_id}/{args.name}_depth_{args.depth}.h5', 'r')
                except Exception:
                    print("Not running as a SLURM job could not move to work")
                    f_depth = h5py.File(f'{name_path}_depth_{args.depth}.h5', 'r')
                mde_runtimes = [f_depth[f'{x}_runtime'][()] / 1e6 for x in image_list]
                if 'completed' not in f_depth:
                    raise ValueError(f'{name_path}_depth_{args.depth}.h5 does not have the completed tag. Aborting.')
            else:
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

                pp_center_1 = np.array(f_images[f'{img_name_1}_size']) / 2
                pp_center_2 = np.array(f_images[f'{img_name_2}_size']) / 2

                if f'{img_name_1}_d' not in f_images:
                    cam1_gt = {'model': 'PINHOLE', 'width': -1, 'height': -1,
                               'params': [K1_gt[0, 0], K1_gt[1, 1], K1_gt[0, 2], K1_gt[1, 2]]}
                else:
                    dist1 = f_images[f'{img_name_1}_d']
                    cam1_gt = {'model': 'OPENCV', 'width': -1, 'height': -1,
                               'params': [K1_gt[0, 0], K1_gt[1, 1], K1_gt[0, 2], K1_gt[1, 2], *dist1]}

                if f'{img_name_2}_d' not in f_images:
                    cam2_gt = {'model': 'PINHOLE', 'width': -1, 'height': -1,
                               'params': [K2_gt[0, 0], K2_gt[1, 1], K2_gt[0, 2], K2_gt[1, 2]]}
                else:
                    dist2 = f_images[f'{img_name_2}_d']
                    cam2_gt = {'model': 'OPENCV', 'width': -1, 'height': -1,
                               'params': [K2_gt[0, 0], K2_gt[1, 1], K2_gt[0, 2], K2_gt[1, 2], *dist2]}

                kps = np.array(f_matches[f"{img_name_1}-{img_name_2}"])

                kp1 = kps[:, :2]
                kp2 = kps[:, 2:4]

                if args.depth != 'gt':
                    depth_map1 = np.array(f_depth[f'{img_name_1}_depth'])
                    depth_map2 = np.array(f_depth[f'{img_name_2}_depth'])

                    d1 = get_kp_depth(kp1, depth_map1, interpolation='nearest')
                    d2 = get_kp_depth(kp2, depth_map2, interpolation='nearest')

                    l = np.logical_and(np.isfinite(d1), np.isfinite(d2))
                    l = np.logical_and(d1 > 0, l)
                    l = np.logical_and(d2 > 0, l)
                    kp1 = kp1[l]
                    kp2 = kp2[l]
                    d1 = d1[l]
                    d2 = d2[l]
                else:
                    d1, d2 = get_gt_depth(kp1, kp2, R_gt, t_gt, K1_gt, K2_gt)

                gt_inlier_mask = get_gt_inlier_mask(kp1, kp2, K1_gt, K2_gt, R_gt, t_gt, args.sampson_threshold)

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
                    for iters in iters_list:
                        yield (experiment, iters, np.copy(kp1), np.copy(kp2), np.copy(d1), np.copy(d2),
                               mde_K1, mde_K2, pp_center_1, pp_center_2, R_gt, t_gt, cam1_gt, cam2_gt,
                               img_name_1, img_name_2, gt_inlier_mask, args.sampson_threshold, args.reprojection_threshold)

        total_length = len(experiments) * len(pair_list) * len(iters_list)

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

        if args.direct_read and f_depth is not None:
            f_depth.close()

        save_full_results(args, full_results)
        save_summary_results(experiments, full_results, mde_runtimes, args)


def get_solvers(args):
    if args.explicit_solvers is not None:
        experiments = args.explicit_solvers.split(',')
        return experiments

    experiments = ['calib']
    if not args.no_mde_K:
        if 'Calib' in args.depth:
            print("Solver using MDE inferred camera params requested, but MDE used GT calibration. Skipping.")
        else:
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
    if not args.no_shift_solvers:
        experiments.extend([f'{x}_shift' for x in experiments])

    if not args.no_reproj_only_ransac:
        experiments.extend([f'{x}_ro' for x in experiments])


    if args.include_baseline_solver:
        experiments.append('baseline_calib')

        if args.include_shared_focal:
            experiments.append('baseline_sf')

        if args.include_varying_focal:
            experiments.append('baseline_vf')


    return experiments


if __name__ == '__main__':
    args = parse_args()
    if args.depth is None:
        mde_list = get_mde_list(args.name, args.data_path)

        for depth_name in mde_list:
            args.depth = depth_name
            print(f"Checking if MDE {depth_name} results are available!")
            h5_path = get_full_results_h5_path(args)
            if os.path.exists(h5_path) and not args.recalc:
                print(f"Results in {h5_path} available. Skipping")
                continue

            print(f"Running for MDE: {depth_name}")

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