import os

import cv2
import h5py
from tqdm import tqdm

from depth_estimators.MoGe import MoGe
from depth_estimators.UniDepth import UniDepth


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Inference script for depth estimation.')
    parser.add_argument('--model_name', type=str, default='MoGeV2', help='Name of the depth estimation model.')
    parser.add_argument('--requires_intrinsics', action='store_true', default=False, help='Whether inference requires intrinsics on input')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu).')
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained model weights.')
    parser.add_argument('out_path', type=str, help='Path to data directory.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset')

    return parser.parse_args()

def get_model(args):
    if args.model_name == 'MoGeV2':
        return MoGe(args.pretrained_weights, version=2, requires_intrinsics=args.requires_intrinsics)
    elif args.model_name == 'MoGeV1':
        return MoGe(args.pretrained_weights, version=1, requires_intrinsics=args.requires_intrinsics)
    elif args.model_name == 'UniDepthV1':
        return UniDepth(args.pretrained_weights, version=1)
    elif args.model_name == 'UniDepthV2':
        return UniDepth(args.pretrained_weights, version=2)
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")


def infer_depth(args):
    model = get_model(args).cuda()

    name_path = os.path.join(args.out_path, args.name)

    image_list_path = f'{name_path}_image_list.txt'
    with open(image_list_path, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]

    f_images = h5py.File(f'{name_path}.h5')
    f_depth = h5py.File(f'{name_path}_depth_{model.name}.h5', 'w')

    for image_name in tqdm(image_list):
        K = f_images[f'{image_name}_K']
        img_path = os.path.join(args.dataset_path, image_name)
        out_dict = model.infer(img_path, K=K)
        f_depth.create_dataset(f'{image_name}_depth', data=out_dict['depth'], compression='gzip', chunks=True)
        if 'inference_K' in out_dict.keys():
            f_depth.create_dataset(f'{image_name}_inference_K', data=out_dict['inference_K'], compression='gzip', chunks=True)
        f_depth.create_dataset(f'{image_name}_K', data=out_dict['K'], compression='gzip', chunks=True)


if __name__ == '__main__':
    args = parse_args()

    infer_depth(args)