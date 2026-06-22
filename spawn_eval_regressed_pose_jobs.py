import argparse
import copy
import os

import submitit

from depth_estimators.infer_depth import get_mde_model
from eval_regressed_pose import REGRESSED_POSE_MODELS, eval_single_model, get_regressed_pose_model_name
from utils.config import config_iterator
from utils.storage import get_full_results_h5_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st', '--sampson_threshold', type=float, default=2.0)
    parser.add_argument('-rt', '--reprojection_threshold', type=float, default=16.0)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('--variance', type=int, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--work_path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--matches', type=str, default='regressed_pose')
    parser.add_argument('dataset_path', nargs='?', default=None)
    parser.add_argument('--account', type=str, default='p1358-25-2',
                        help='Slurm account name')
    parser.add_argument('--queue', type=str, default='short',
                        help='Slurm partition/queue name')
    parser.add_argument('--mem_gb', type=int, default=64,
                        help='Memory per job in GB')
    parser.add_argument('--timeout_min', type=int, default=60,
                        help='Expected max runtime per job in minutes')
    parser.add_argument('--cpus_per_task', type=int, default=4)
    parser.add_argument('--gpus_per_node', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for submitit logs (default: <work_path>/slurm_logs)')
    return parser.parse_args()


def _model_specs(args):
    if args.model_name is not None:
        return [(args.model_name, args.pretrained_weights)]
    return [(model_name, weights)
            for model_name, weight_list in REGRESSED_POSE_MODELS.items()
            for weights in weight_list]


def run_for_model(args, model_name, weights):
    model = get_mde_model(model_name, weights)
    eval_single_model(model, args)


def main(args):
    args.full_results_type = 'regression_full'
    log_dir = args.log_dir or os.path.join(args.work_path, 'slurm_logs')
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=args.account,
        slurm_partition=args.queue,
        mem_gb=args.mem_gb,
        timeout_min=args.timeout_min,
        cpus_per_task=args.cpus_per_task,
        gpus_per_node=args.gpus_per_node,
    )

    array_args = []
    for model_name, weights in _model_specs(args):
        job_args = copy.copy(args)
        job_args.depth = get_regressed_pose_model_name(model_name, weights)
        h5_path = get_full_results_h5_path(job_args)
        if os.path.exists(h5_path) and not args.recalc and not args.load:
            print(f"Results for {job_args.depth} already available at {h5_path}. Skipping.")
            continue
        array_args.append((copy.copy(args), model_name, weights))

    jobs = executor.map_array(run_for_model, *zip(*array_args)) if array_args else []

    print(f"\nSubmitted {len(jobs)} job(s) with log dir {log_dir}:")
    for (job_args, model_name, weights), job in zip(array_args, jobs):
        print(f"Model: {get_regressed_pose_model_name(model_name, weights)} job_id={job.job_id}")


if __name__ == '__main__':
    args = parse_args()
    if args.config_path is not None:
        for name, config in config_iterator(args.config_path):
            single_args = copy.copy(args)
            single_args.name = name
            single_args.work_path = config['work_path']
            single_args.dataset_path = config.get('dataset_path', config['path'])
            main(single_args)
    else:
        main(args)
