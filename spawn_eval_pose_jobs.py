import argparse
import copy
import json
import os
import submitit

from eval_pose import eval_single_mde
from utils.results import get_basename, get_mde_list
from utils.storage import get_full_results_h5_path


def parse_args():
    parser = argparse.ArgumentParser()
    # --- same args as eval_pose.py ---
    parser.add_argument('-st',  '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-sw',  '--sampson_weight', type=float, default=1.0)
    parser.add_argument('-rt',  '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('-nmk', '--no_mde_K', action='store_true', default=False)
    parser.add_argument('-bs',  '--include_baseline_solver', action='store_true', default=False)
    parser.add_argument('-nss', '--no_shift_solvers', action='store_true', default=False)
    parser.add_argument('-sf',  '--include_shared_focal', action='store_true', default=False)
    parser.add_argument('-vf',  '--include_varying_focal', action='store_true', default=False)
    parser.add_argument('-dr', '--direct_read', action='store_true', default=False)
    parser.add_argument('-nro', '--no_reproj_only_ransac', action='store_true', default=False)
    parser.add_argument('--timeout_pool', action='store_true', default=False)
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('--explicit_solvers', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--work_path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--matches', type=str, default='splg_2048_noresize')
    # --- slurm-specific args ---
    parser.add_argument('--account', type=str, default='p1358-25-2',
                        help='Slurm account name')
    parser.add_argument('--queue', type=str, default='short',
                        help='Slurm partition/queue name')
    parser.add_argument('--mem_gb', type=int, default=32,
                        help='Memory per job in GB')
    parser.add_argument('--timeout_min', type=int, default=20,
                        help='Expected max runtime per job in minutes')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for submitit logs (default: <work_path>/slurm_logs)')
    return parser.parse_args()

def run_for_depth(args):
    if args.depth == 'gt':
        args.include_baseline_solver = True
    eval_single_mde(args)


def main(args):
    log_dir = args.log_dir or os.path.join(args.work_path, 'slurm_logs')
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=args.account,
        slurm_partition=args.queue,
        mem_gb=args.mem_gb,
        timeout_min=args.timeout_min,
        cpus_per_task=args.num_workers
    )

    mde_list = get_mde_list(args.name, args.work_path)

    depths_to_run = ['gt'] + mde_list

    array_job_arguments = []

    for depth_name in depths_to_run:
        job_args = copy.copy(args)
        job_args.depth = depth_name
        h5_path = get_full_results_h5_path(job_args)
        if os.path.exists(h5_path) and not args.recalc:
            print(f"Results for {depth_name} already available at {h5_path}. Skipping.")
            continue
        job_args.depth = depth_name
        array_job_arguments.append(job_args)

    jobs = executor.map_array(run_for_depth, array_job_arguments)

    print(f"\nSubmitted {len(jobs)} job(s):")
    for depth_name, job in zip(depths_to_run, jobs):
        print(f"Depth: {depth_name} job_id={job.job_id}")


if __name__ == '__main__':
    args = parse_args()
    if args.config_path is not None:
        with open(args.config_path) as f:
            dataset_config = json.load(f)

        for name, config in dataset_config.items():
            single_args = copy.copy(args)
            single_args.name = name
            single_args.work_path = config["work_path"]
            single_args.direct_read = "requires_direct_read" in config.keys() and config["requires_direct_read"]

            if "cameras" in config:
                if config["cameras"] == 'shared':
                    single_args.include_shared_focal = True
                    single_args.include_varying_focal = False
                if config["cameras"] == 'varying':
                    single_args.include_shared_focal = False
                    single_args.include_varying_focal = True

            main(single_args)
    else:
        main(args)
