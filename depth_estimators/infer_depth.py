import os
from pathlib import Path, PureWindowsPath

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from depth_estimators.DepthAnything import DepthAnything
from depth_estimators.MoGe import MoGe
from depth_estimators.UniDepth import UniDepth
from utils.system_info import save_metadata

ALL_MDEs = {
    'DepthAnythingV3Calib': ['DA3METRIC-LARGE', 'DA3MONO-LARGE'],
    'DepthAnythingV3': ['DA3METRIC-LARGE', 'DA3MONO-LARGE'],
    'MoGeV1': ['moge-vitl'],
    'MoGeV2': ['moge-2-vitl'],
    'MoGeV1Calib': ['moge-vitl'],
    'MoGeV2Calib': ['moge-2-vitl'],
    'UniDepthV2': ['vits14', 'vitb14', 'vitl14'],
    'UniDepthV1': ['vitl14', 'v1-cnvnxtl']
    }




def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Inference script for depth estimation.')
    parser.add_argument('--model_name', type=str, default='all', help='Name of the depth estimation model.')
    parser.add_argument('--recalc', action='store_true', default=False, help='Whether inference requires intrinsics on input')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu).')
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained model weights.')
    parser.add_argument('out_path', type=str, help='Path to data directory.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset')

    return parser.parse_args()

def get_mde_model(model_name, weights):
    if model_name == 'MoGeV2':
        return MoGe(weights, version=2, requires_intrinsics=False)
    elif model_name == 'MoGeV2Calib':
        return MoGe(weights, version=2, requires_intrinsics=True)

    elif model_name == 'MoGeV1':
        return MoGe(weights, version=1, requires_intrinsics=False)
    elif model_name == 'MoGeV1Calib':
        return MoGe(weights, version=1, requires_intrinsics=True)

    elif model_name == 'UniDepthV1':
        return UniDepth(weights, version=1)
    elif model_name == 'UniDepthV2':
        return UniDepth(weights, version=2)

    elif model_name == 'DepthAnythingV3':
        return DepthAnything(weights, version=3)
    elif model_name == 'DepthAnythingV3Calib':
        return DepthAnything(weights, version=3, requires_intrinsics=True)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


def infer_depth(model, args):
    name_path = os.path.join(args.out_path, args.name)

    f_depth_path = f'{name_path}_depth_{model.name}.h5'
    if os.path.exists(f_depth_path) and not args.recalc:
        try:
            with h5py.File(f_depth_path, 'r') as f_check:
                if 'completed' in f_check:
                    print(f"File {f_depth_path} already exists and is completed. Skipping. Use --recalc arg to force recalculation.")
                    return
                print(f"File {f_depth_path} exists but is incomplete. Recalculating...")
        except Exception:
            print(f"File {f_depth_path} exists but is broken. Recalculating...")

    print("Loading Model")
    model.load_model()

    print(f"Creating {f_depth_path}")
    f_images = h5py.File(f'{name_path}.h5', 'r')

    f_depth = h5py.File(f_depth_path, 'w')
    save_metadata(f_depth)

    image_list_path = f'{name_path}_image_list.txt'
    with open(image_list_path, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]


    for image_name in tqdm(image_list):
        K = np.array(f_images[f'{image_name}_K'])
        img_path = os.path.join(args.dataset_path, Path(PureWindowsPath(image_name)))
        size = np.array(f_images[f'{image_name}_size'])
        size_orig = np.array(f_images[f'{image_name}_size_orig'])
        if (size != size_orig).any():
            # if we resized during dataset generation we need to resize here
            out_dict = model.infer(img_path, size=size, K=K)
        else:
            out_dict = model.infer(img_path, K=K)

        f_depth.create_dataset(f'{image_name}_depth', data=out_dict['depth'].astype(np.float16), compression='gzip', chunks=True)
        f_depth.create_dataset(f'{image_name}_runtime', data=out_dict['runtime'])

        if 'inference_K' in out_dict.keys():
            f_depth.create_dataset(f'{image_name}_inference_K', data=out_dict['inference_K'], compression='gzip', chunks=True)
        if 'K' in out_dict.keys():
            f_depth.create_dataset(f'{image_name}_K', data=out_dict['K'], compression='gzip', chunks=True)

    f_depth.create_group('completed')
    print(f"{f_depth} completed")
    f_images.close()
    f_depth.close()


def run(args):
    if args.model_name == 'all':
        failed_models = []

        for model_name, weight_list in ALL_MDEs.items():
            for weights in weight_list:

                print(f"Running for model {model_name} with weights {weights}")
                model = get_mde_model(model_name, weights)
                try:
                    infer_depth(model, args)
                except Exception as e:
                    print(f"Model {model_name} with weights {weights} failed with error: {e}")
                    failed_models.append((model_name, weights))

        print(f'Inference failed for some cases: {failed_models}')
    else:
        model = get_mde_model(args.model_name, args.pretrained_weights)
        infer_depth(model, args)

if __name__ == '__main__':
    args = parse_args()
    run(args)