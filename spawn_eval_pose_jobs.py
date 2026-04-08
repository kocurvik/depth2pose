import argparse
import copy
import os
import submitit

from eval_pose import eval_single_mde
from utils.results import get_basename, get_full_results_h5_path


def parse_args():
    parser = argparse.ArgumentParser()
    # --- same args as eval_pose.py ---
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
    parser.add_argument('--explicit_solvers', type=str, default=None)
    parser.add_argument('data_path')
    parser.add_argument('name')
    parser.add_argument('matches')
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
                        help='Directory for submitit logs (default: <data_path>/slurm_logs)')
    return parser.parse_args()

def run_for_depth(args):
    if args.depth == 'gt':
        args.include_baseline_solver = True
    eval_single_mde(args)


def main():
    args = parse_args()

    log_dir = args.log_dir or os.path.join(args.data_path, 'slurm_logs')
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=args.account,
        slurm_partition=args.queue,
        mem_gb=args.mem_gb,
        timeout_min=args.timeout_min,
        cpus_per_task=args.num_workers
    )

    mde_list = [
        x.split('_depth_')[1].split('.h5')[0]
        for x in os.listdir(args.data_path)
        if x.startswith(f'{args.name}_depth_') and x.endswith('.h5')
    ]

    depths_to_run = []

    for depth_name in mde_list:
        h5_path = os.path.join(args.data_path, f'full_results/{get_basename(args, depth_name)}.h5')
        if os.path.exists(h5_path) and not args.recalc:
            print(f"Results for {depth_name} already available at {h5_path}. Skipping.")
            continue
        depths_to_run.append(depth_name)

    depths_to_run.append('gt')

    array_job_arguments = []

    for depth_name in mde_list:
        job_args = copy.copy(args)
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
    main()