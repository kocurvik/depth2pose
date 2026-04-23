import argparse
import copy
import json
import os
import shutil
import h5py
import submitit

from depth_estimators.infer_depth import ALL_MDEs, get_mde_model, infer_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Spawn Slurm jobs for depth inference.')
    # --- same args as infer_depth.py ---
    parser.add_argument('--recalc', action='store_true', default=False,
                        help='Force recalculation even if output already exists')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('--temp_out_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, help='Path to output directory')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    # --- slurm-specific args ---
    parser.add_argument('--config_path', type=str, default=None,
                        help='specify path to config to run for multiple datasets at the same time')
    parser.add_argument('--account', type=str, default='p1358-25-2',
                        help='Slurm account name')
    parser.add_argument('--queue', type=str, default='gpu',
                        help='Slurm partition/queue name')
    parser.add_argument('--mem_gb', type=int, default=32,
                        help='Memory per job in GB')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Memory per job in GB')
    parser.add_argument('--timeout_min', type=int, default=240,
                        help='Expected max runtime per job in minutes')
    parser.add_argument('--gpus_per_node', type=int, default=1,
                        help='Number of GPUs per job')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for submitit logs (default: <out_path>/slurm_logs)')
    parser.add_argument('--work_dir', action='store_true', default=False,
                        help='If set, write h5 output to /work/$SLURM_JOB_ID/ during inference '
                             'and move to out_path when done')
    return parser.parse_args()


def run_for_model(args):
    model = get_mde_model(args.model_name, args.pretrained_weights)

    if args.work_dir:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        # array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        # if array_task_id:
        #     job_id = f'{job_id}_{array_task_id}'
        tmp_out_path = f'/work/{job_id}/'
        os.makedirs(tmp_out_path, exist_ok=True)

        final_out_path = args.out_path
        args = copy.copy(args)
        args.temp_out_path = tmp_out_path
        print(f"Working in {tmp_out_path}")

        infer_depth(model, args)

        name_path = os.path.join(tmp_out_path, args.name)
        src = f'{name_path}_depth_{model.name}.h5'
        print(f"Moving back from {src} to {final_out_path}")
        os.makedirs(final_out_path, exist_ok=True)
        shutil.move(src, final_out_path)
    else:
        infer_depth(model, args)


def main(args):
    log_dir = args.log_dir or os.path.join(args.out_path, 'slurm_logs')
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

    name_path = os.path.join(args.out_path, args.name)
    array_job_arguments = []

    for model_name, weight_list in ALL_MDEs.items():
        for weights in weight_list:
            model = get_mde_model(model_name, weights)
            f_depth_path = f'{name_path}_depth_{model.name}.h5'
            if os.path.exists(f_depth_path) and not args.recalc:
                try:
                    with h5py.File(f_depth_path, 'r') as f_check:
                        if 'completed' in f_check:
                            print(f"Output for {model_name} ({weights}) already complete at {f_depth_path}. Skipping.")
                            continue
                        print(f"Output for {model_name} ({weights}) exists but is incomplete. Resubmitting.")
                except Exception:
                    print(f"Output for {model_name} ({weights}) exists but is broken. Resubmitting.")

            job_args = copy.copy(args)
            job_args.model_name = model_name
            job_args.pretrained_weights = weights
            array_job_arguments.append(job_args)

    jobs = executor.map_array(run_for_model, array_job_arguments)

    print(f"\nSubmitted {len(jobs)} job(s):")
    for job_args, job in zip(array_job_arguments, jobs):
        print(f"Model: {job_args.model_name} weights: {job_args.pretrained_weights} job_id={job.job_id}")


if __name__ == '__main__':
    args = parse_args()
    if args.config_path is not None:
        with open(args.config_path) as f:
            dataset_config = json.load(f)

        for name, config in dataset_config.items():
            single_args = copy.copy(args)
            single_args.name = name
            single_args.out_path = config["work_path"]
            single_args.dataset_path = config["path"]
            main(single_args)
    else:
        main(args)