import argparse
import copy
import json
import os
import shutil
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
import submitit
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from tqdm import tqdm

from utils.system_info import save_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_features', type=int, default=2048)
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


def extract_features(args):
    name_path = os.path.join(args.out_path, args.name)

    cache_path = f'{name_path}-extracted_sp_{args.max_features}_{args.resize if args.resize is not None else "noresize"}.pt'

    if os.path.exists(cache_path) and not args.recalc_features:
        print(f"Features already found in {cache_path}")
        return torch.load(cache_path)

    image_list_path = f'{name_path}_image_list.txt'
    with open(image_list_path, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]

    with h5py.File(f'{name_path}.h5', 'r') as f_images:
        f_images = {key: np.array(f_images[key]) for key in f_images.keys()}

    extractor = SuperPoint(max_num_keypoints=args.max_features).eval().cuda()

    print("Extracting features")
    feature_dict = {}

    for img_name in tqdm(image_list):
        img_path = os.path.join(args.dataset_path, Path(PureWindowsPath(img_name)))

        size = np.array(f_images[f'{img_name}_size'])
        size_orig = np.array(f_images[f'{img_name}_size_orig'])

        if (size != size_orig).any():
            # if we resized during dataset creation and call this without specific resizing then we need to resize here
            image_tensor = load_image(img_path, resize=tuple(size[::-1])).cuda()
        else:
            image_tensor = load_image(img_path).cuda()

        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[img_name] = kp_tensor

    f_images.close()

    torch.save(feature_dict, cache_path)
    print("Features saved to: ", cache_path)

    return feature_dict


def match_pairs(feature_dict, args):
    name_path = os.path.join(args.out_path, args.name)

    image_list_path = f'{name_path}_image_pairs.txt'
    with open(image_list_path, 'r') as f:
        pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

    matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()

    main_h5_path = f'{name_path}_splg_{args.max_features}_{args.resize if args.resize is not None else "noresize"}.h5'

    if args.work_dir:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        tmp_out_path = f'/work/{job_id}/'
        os.makedirs(tmp_out_path, exist_ok=True)
        h5_path = os.path.join(tmp_out_path, f'{args.name}_splg_{args.max_features}_{args.resize if args.resize is not None else "noresize"}.h5')
    else:
        h5_path = main_h5_path

    with h5py.File(h5_path, 'w') as f:
        save_metadata(f)
        print("Matching features")
        with torch.no_grad():
            for img_name_1, img_name_2 in tqdm(pair_list):
                feats_1 = feature_dict[img_name_1]
                feats_2 = feature_dict[img_name_2]
                out = matcher({'image0': feats_1, 'image1': feats_2})
                feats_1, feats_2, out = [rbd(x) for x in [feats_1, feats_2, out]]  # remove batch dimension
                matches = out['matches']
                score = out['scores']
                points_1 = feats_1['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
                points_2 = feats_2['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

                kpts_1 = points_1.cpu().numpy()
                kpts_2 = points_2.cpu().numpy()
                score = score.cpu().numpy()

                match_positions = np.hstack([kpts_1[:, :2], kpts_2[:, :2], score[:, np.newaxis]])
                f.create_dataset(f"{img_name_1}-{img_name_2}", data=match_positions, compression='gzip', chunks=True)
            f.create_group("completed")

    if args.work_dir:
        shutil.move(h5_path, main_h5_path)


def match_splg(args):
    feature_dict = extract_features(args)
    match_pairs(feature_dict, args)


if __name__ == '__main__':
    args = parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            dataset_config = json.load(f)

        job_args = []

        for name, config in dataset_config.items():
            single_args = copy.copy(args)
            single_args.name = name
            single_args.out_path = config["work_path"]
            single_args.dataset_path = config["path"]

            name_path = os.path.join(single_args.out_path, single_args.name)
            h5_path = f'{name_path}_splg_{single_args.max_features}_{single_args.resize if args.resize is not None else "noresize"}.h5'
            if os.path.exists(h5_path):
                with h5py.File(h5_path,'r') as f:
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

            job = executor.submit(match_splg, single_args)
            print(f"Sumitted job id {job.job_id} for:", single_args.dataset_path)

    else:
        match_splg(args)
