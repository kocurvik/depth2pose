import argparse
import copy
import os
import shutil
from pathlib import PureWindowsPath, Path

import h5py
import numpy as np
import submitit
from tqdm import tqdm

import torch

from utils.config import config_iterator
from utils.system_info import save_metadata

# os.environ["TORCH_COMPILE_DISABLE"] = "1"

from romav2 import RoMaV2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_features', type=int, default=2048*8)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('--recalc_features', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)

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


def match_pairs(args):
    name_path = os.path.join(args.out_path, args.name)

    image_list_path = f'{name_path}_image_pairs.txt'
    with open(image_list_path, 'r') as f:
        pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

    # load pretrained model
    model = RoMaV2().eval().cuda()
    model.apply_setting('base')

    main_h5_path = f'{name_path}_roma_v2_base_{args.max_features}.h5'

    if args.work_dir:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        tmp_out_path = f'/work/{job_id}/'
        os.makedirs(tmp_out_path, exist_ok=True)
        h5_path = os.path.join(tmp_out_path, f'{args.name}_roma_v2_base_{args.max_features}.h5')
    else:
        h5_path = main_h5_path

    with h5py.File(h5_path, 'w') as f, h5py.File(f'{name_path}.h5', 'r') as f_images:
        save_metadata(f)
        print("Matching features")
        with torch.no_grad():
            for img_name_1, img_name_2 in tqdm(pair_list):
                img_path_1 = os.path.join(args.dataset_path, Path(PureWindowsPath(img_name_1)))
                img_path_2 = os.path.join(args.dataset_path, Path(PureWindowsPath(img_name_2)))

                preds = model.match(img_path_1, img_path_2)

                # you can also run the forward method directly as
                # preds = model(img_A, img_B)

                # Sample 5000 matches for estimation
                matches, overlaps, precision_AB, precision_BA = model.sample(preds, args.max_features)

                W_A, H_A = f_images[f'{img_name_1}_size']
                W_B, H_B = f_images[f'{img_name_2}_size']
                kpts_1, kpts_2 = model.to_pixel_coordinates(matches.cpu(), H_A, W_A, H_B, W_B)

                # kpts_1 = kpts_1.numpy()
                # kpts_2 = kpts_2.numpy()
                # score = score.cpu().numpy()

                match_positions = np.hstack([kpts_1[:, :2], kpts_2[:, :2]])
                f.create_dataset(f"{img_name_1}-{img_name_2}", data=match_positions, compression='gzip', chunks=True)
        f.create_group("completed")

    if args.work_dir:
        shutil.move(h5_path, main_h5_path)


def match_roma(args):
    match_pairs(args)


if __name__ == '__main__':
    args = parse_args()

    if args.config_path is not None:
        for name, config in config_iterator(args.config_path):
            single_args = copy.copy(args)
            single_args.name = name
            single_args.out_path = config["work_path"]
            single_args.dataset_path = config["path"]

            name_path = os.path.join(single_args.out_path, single_args.name)
            h5_path = f'{name_path}_roma_v2_base_{args.max_features}.h5'
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    if 'completed' in f.keys():
                        print(f"Skippting since h5py: {h5_path} already contains completed")
                        continue

            executor = submitit.AutoExecutor(folder=os.path.join(config["work_path"], "slurm_logs"))
            executor.update_parameters(
                slurm_account=args.account,
                slurm_partition=args.queue,
                mem_gb=args.mem_gb,
                timeout_min=args.timeout_min,
                gpus_per_node=args.gpus_per_node,
                cpus_per_task=args.num_workers
            )

            job = executor.submit(match_roma, single_args)
            print(f"Sumitted job id {job.job_id} for:", single_args.dataset_path)

    else:
        match_roma(args)