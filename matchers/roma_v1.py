import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

import torch

from utils.system_info import save_metadata

os.environ["TORCH_COMPILE_DISABLE"] = "1"

# from romav2 import RoMaV2
from romatch import roma_outdoor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=2048)
    parser.add_argument('--recalc_features', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def match_roma(args):
    name_path = os.path.join(args.out_path, args.name)

    image_list_path = f'{name_path}_image_pairs.txt'
    with open(image_list_path, 'r') as f:
        pair_list = [x.strip().split(',')[:2] for x in f.readlines()]

    # load pretrained model
    # model = RoMaV2().eval().cuda()
    # model.apply_setting('base')

    roma_model = roma_outdoor(device='cuda', use_custom_corr=False).eval()

    f = h5py.File(f'{name_path}_roma_v1_{args.max_features}.h5','w')
    save_metadata(f)

    f_images = h5py.File(f'{name_path}.h5')

    print("Matching features")
    with torch.no_grad():
        for img_name_1, img_name_2 in tqdm(pair_list):
            img_path_1 = os.path.join(args.dataset_path, img_name_1)
            img_path_2 = os.path.join(args.dataset_path, img_name_2)

            W_A, H_A = f_images[f'{img_name_1}_size']
            W_B, H_B = f_images[f'{img_name_2}_size']

            warp, certainty = roma_model.match(img_path_1, img_path_2, device='cuda')
            matches, certainty = roma_model.sample(warp, certainty)
            kpts_1, kpts_2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

            # roma v2
            # preds = model.match(img_path_1, img_path_2)
            # matches, overlaps, precision_AB, precision_BA = model.sample(preds, args.max_features)
            # kpts_1, kpts_2 = model.to_pixel_coordinates(matches.cpu(), H_A, W_A, H_B, W_B)

            kpts_1 = kpts_1.cpu().numpy()
            kpts_2 = kpts_2.cpu().numpy()
            score = certainty.cpu().numpy()

            match_positions = np.hstack([kpts_1[:, :2], kpts_2[:, :2], score[:, np.newaxis]])
            f.create_dataset(f"{img_name_1}-{img_name_2}", data=match_positions, compression='gzip', chunks=True)
    f_images.close()
    f.close()


if __name__ == '__main__':
    args = parse_args()
    match_roma(args)
