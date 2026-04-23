import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import h5py
import numpy as np
import submitit
import torch
from sympy.physics.units import action
from torch.backends.cudnn import benchmark
from tqdm import tqdm
from collections import defaultdict

from utils.results import get_mde_list

sys.path.insert(0, "./")
from datasets.dataloader import EvalDataLoaderPipeline
from utils.metrics import DepthMetrics, key_average


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu to be used for extraction')
    parser.add_argument('--recalc', action='store_true', default = False,
                        help='whether to recalculate even previously calculated depths')

    parser.add_argument('--account', type=str, default='p1358-25-2',
                        help='Slurm account name')
    parser.add_argument('--queue', type=str, default='gpu',
                        help='Slurm partition/queue name')
    parser.add_argument('--mem_gb', type=int, default=32,
                        help='Memory per job in GB')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Memory per job in GB')
    parser.add_argument('--timeout_min', type=int, default=60,
                        help='Expected max runtime per job in minutes')
    parser.add_argument('--gpus_per_node', type=int, default=1,
                        help='Number of GPUs per job')
    parser.add_argument('--work_dir', action='store_true', default=False,
                        help='If set, write h5 output to /work/$SLURM_JOB_ID/ during inference '
                             'and move to out_path when done')

    return parser.parse_args()


def get_depth_from_h5(f_depth_h5, scene_name, file_name):
    depth_key_name = f"{scene_name}\\images\\{file_name}_depth"
    depth = np.array(f_depth_h5[depth_key_name])
    depth[depth <= 0] = np.inf
    return depth

def evaluate_model(mde_model, benchmark_name, benchmark_config, device, use_work_dir=False, recalc=False):
    metric_fn = DepthMetrics()

    if 'contains_gt_depth' not in benchmark_config or not benchmark_config['contains_gt_depth']:
        return
    single_results_path = Path(benchmark_config['work_path']) / 'depth_results' / f'{mde_model}.json'

    if os.path.exists(single_results_path) and not recalc:
        print(f"{single_results_path} exists, skipping")
        return
    else:
        h5_depth_path = Path(benchmark_config['work_path']) / f'{benchmark_name}_depth_{mde_model}.h5'
        if use_work_dir:
            job_id = os.environ.get('SLURM_JOB_ID', 'local')
            work_dir = f'/work/{job_id}'
            tmp_path = Path(work_dir) / f'{benchmark_name}_depth_{mde_model}.h5'
            print(f"Copying {h5_depth_path} to {tmp_path}")
            shutil.copy(h5_depth_path, tmp_path)
            h5_depth_path = tmp_path

        metric_result_list, scale_result_list, affine_result_list = [], [], []
        with (
            EvalDataLoaderPipeline(benchmark_config['path'], benchmark_config['work_path'],
                                   width=benchmark_config['width'], height=benchmark_config['height'],
                                   depth_unit=benchmark_config['depth_unit']) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar,
            h5py.File(h5_depth_path,
                      'r') as f_depth_h5
        ):
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                }
                scenename, filename = sample["scenename"], sample["filename"]
                _, gt_depth, depth_mask = (
                    sample["image"],
                    sample["depth"],
                    sample["depth_mask"]
                )
                # pd_depth_data = np.load(eval_data_pipe.path / scenename / mde_model / f"{filename}.npz")
                # pd_depth, pd_intrinsic = pd_depth_data["depth"], pd_depth_data["K"]

                pd_depth = get_depth_from_h5(f_depth_h5, scenename, filename)
                pd_depth = torch.from_numpy(pd_depth).to(torch.float32).to(device)

                metric_results = metric_fn.compute_metric_depth(gt_depth, pd_depth, depth_mask)
                scale_results = metric_fn.compute_scale_inv_depth(gt_depth, pd_depth, depth_mask)
                affine_results = metric_fn.compute_affine_inv_depth(gt_depth, pd_depth, depth_mask)
                metric_result_list.append(metric_results)
                scale_result_list.append(scale_results)
                affine_result_list.append(affine_results)

                pbar.update(1)

        single_results = {'metric': key_average(metric_result_list), 'scale': key_average(scale_result_list),
                          'affine': key_average(affine_result_list)}

        Path(single_results_path).parent.mkdir(exist_ok=True, parents=True)
        Path(single_results_path).write_text(json.dumps(single_results, indent=4))


def get_depth_models(processed_dataset_path):
    depth_models = sorted(list((Path(processed_dataset_path)).glob("*.h5")))
    depth_models = [depth_model.stem.split("_depth_")[-1] for depth_model in depth_models]
    return depth_models

def main():
    args = parse_args()
    config_path = args.config_path

    with open(config_path) as f:
        dataset_config = json.load(f)

    device = torch.device(args.device)

    first_dataset_name = list(dataset_config.keys())[0]
    depth_models = get_mde_list(first_dataset_name, dataset_config[first_dataset_name]['work_path'])

    job_args = []

    for benchmark_name, benchmark_config in dataset_config.items():
        if 'contains_gt_depth' not in benchmark_config or not benchmark_config['contains_gt_depth']:
            continue

        for mde_model in depth_models:
            single_results_path = Path(benchmark_config['work_path']) / 'depth_results' / f'{mde_model}.json'

            if os.path.exists(single_results_path) and not args.recalc:
                print(f"Skipping: {benchmark_name} - {mde_model} since the results already exists in {single_results_path}")

            job_args.append((mde_model, benchmark_name, benchmark_config, device, args.work_dir, args.recalc))

        log_dir = args.log_dir or os.path.join(benchmark_config['work_path'], 'slurm_logs')
        os.makedirs(log_dir, exist_ok=True)

        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(
            slurm_account=args.account,
            slurm_partition=args.queue,
            mem_gb=args.mem_gb,
            timeout_min=args.timeout_min,
            gpus_per_node=args.gpus_per_node,
            cpus_per_task=args.num_workers
        )

        jobs = executor.map_array(evaluate_model, job_args)

        print(f"\nSubmitted {len(jobs)} job(s):")
        for job_arg, job in zip(job_args, jobs):
            print(f"Model: {job_arg[0]} dataset: {job_arg[1]} job_id={job.job_id}")


if __name__ == "__main__":
    main()
