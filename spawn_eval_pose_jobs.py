import argparse
import copy
import os
import submitit

from eval_pose import eval_single_mde


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


def run_for_depth(args, depth: str, force_baseline: bool = False):
    job_args = copy.copy(args)
    job_args.depth = depth
    if force_baseline:
        job_args.include_baseline_solver = True
    eval_single_mde(job_args)


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
        cpus_per_task=args.num_workers,
    )

    mde_list = [
        x.split('_depth_')[1].split('.h5')[0]
        for x in os.listdir(args.data_path)
        if x.startswith(f'{args.name}_depth_') and x.endswith('.h5')
    ]

    depths_to_run = []

    for depth_name in mde_list:
        basename = (f'{args.name}_{args.matches}_{depth_name}'
                    f'_{args.sampson_threshold}t_{args.reprojection_threshold}r')
        h5_path = os.path.join(args.data_path, f'full_results/{basename}.h5')
        if os.path.exists(h5_path) and not args.recalc:
            print(f"Results for {depth_name} already available at {h5_path}. Skipping.")
            continue
        depths_to_run.append(depth_name)

    depths_to_run.append('gt')

    with executor.batch():
        jobs = []
        for depth_name in depths_to_run:
            force_baseline = depth_name == 'gt'
            basename = (f'{args.name}_{args.matches}_{depth_name}'
                        f'_{args.sampson_threshold}t_{args.reprojection_threshold}r')
            executor.update_parameters(
                slurm_additional_parameters={
                    'output': os.path.join(log_dir, f'{basename}_%j.out'),
                    'error':  os.path.join(log_dir, f'{basename}_%j.err'),
                }
            )
            print(f"Queuing job for depth: {depth_name}")
            job = executor.submit(run_for_depth, args, depth_name, force_baseline)
            jobs.append((depth_name, job))

    print(f"\nSubmitted {len(jobs)} job(s):")
    for depth_name, job in jobs:
        print(f"  depth={depth_name}  job_id={job.job_id}")


if __name__ == '__main__':
    main()