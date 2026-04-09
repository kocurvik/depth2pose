import os
from pathlib import Path, PureWindowsPath
import json
import h5py
import numpy as np
import traceback

from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True  # Pick fastest kernels
import sys
sys.path.insert(0, "./")
from depth_estimators.Metric3D import Metric3D
from depth_estimators.DepthAnything import DepthAnything
from depth_estimators.MoGe import MoGe
from depth_estimators.UniDepth import UniDepth
from depth_estimators.UniK3D import UniK3D
from depth_estimators.DepthPro import DepthPro
from utils.system_info import save_metadata

ALL_MDEs = {
    'UniK3D': ['vitl'],
    'UniK3DCalib': ['vitl'],
    'Metric3DV2': ['vit_small', 'vit_large', 'vit_giant2'],
    'DepthAnythingV2': ['vits', 'vitb', 'vitl'],
    'DepthAnythingV3': ['DA3METRIC-LARGE', 'DA3MONO-LARGE'],
    'DepthAnythingV3Calib': ['DA3METRIC-LARGE', 'DA3MONO-LARGE'],
    'MoGeV1': ['moge-vitl'],
    'MoGeV2': ['moge-2-vitl'],
    'MoGeV1Calib': ['moge-vitl'],
    'MoGeV2Calib': ['moge-2-vitl'],
    'UniDepthV2': ['vits14', 'vitb14', 'vitl14'],
    'UniDepthV2Calib': ['vits14', 'vitb14', 'vitl14'],
    'UniDepthV1': ['vitl14', 'cnvnxtl'],
    'UniDepthV1Calib': ['vitl14', 'cnvnxtl'],
    'DepthPro': ['vitl'],
    'DepthProCalib': ['vitl'],
    }




def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Inference script for depth estimation.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the depth estimation model.')
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

    elif model_name == 'Metric3DV2':
        return Metric3D(weights, version=2, requires_intrinsics=False)

    elif model_name == 'MoGeV1':
        return MoGe(weights, version=1, requires_intrinsics=False)
    elif model_name == 'MoGeV1Calib':
        return MoGe(weights, version=1, requires_intrinsics=True)

    elif model_name == 'UniK3D':
        return UniK3D(weights, version=1, requires_intrinsics=False)
    elif model_name == 'UniK3DCalib':
        return UniK3D(weights, version=1, requires_intrinsics=True)

    elif model_name == 'DepthPro':
        return DepthPro(weights, version=1, requires_intrinsics=False)
    elif model_name == 'DepthProCalib':
        return DepthPro(weights, version=1, requires_intrinsics=True)

    elif model_name == 'UniDepthV1':
        return UniDepth(weights, version=1)
    elif model_name == 'UniDepthV2':
        return UniDepth(weights, version=2)

    elif model_name == 'UniDepthV1Calib':
        return UniDepth(weights, version=1, requires_intrinsics=True)
    elif model_name == 'UniDepthV2Calib':
        return UniDepth(weights, version=2, requires_intrinsics=True)

    elif model_name == 'DepthAnythingV2':
        return DepthAnything(weights, version=2)
    elif model_name == 'DepthAnythingV3':
        return DepthAnything(weights, version=3)
    elif model_name == 'DepthAnythingV3Calib':
        return DepthAnything(weights, version=3, requires_intrinsics=True)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


def infer_depth(model, args):
    dataset_root = Path(args.dataset_path)
    out_dir = Path(args.out_path)

    print("Loading Model")
    model.load_model()

    scenes = sorted(list(dataset_root.glob("*")))
    scene_names = [scene.stem for scene in scenes if scene.is_dir()]

    for scene_name in tqdm(scene_names):
        filenames = sorted(list((dataset_root / scene_name / "images").glob("*.jpg")))
        filenames = [file.stem for file in filenames]
        for filename in filenames:
            img_path = dataset_root / scene_name / "images" / f"{filename}.jpg"
            intrinsic_path = dataset_root / scene_name / "intrinsics" / f"{filename}.json"
            with open(intrinsic_path) as f:
                K = json.load(f)['intrinsics']
                K = np.array(K)
                K[0, :] *= 2048
                K[1, :] *= 1365
            out_dict = model.infer(img_path, K=K)
            depth, K, runtime = out_dict['depth'], out_dict['inference_K'], out_dict['runtime']
            (out_dir / scene_name / f"{model.name}").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / scene_name / f"{model.name}" / f"{filename}", 
                                depth=depth.astype(np.float16),
                                runtime=runtime,
                                K=K)
        # break


def run(args):
    if args.model_name is None:
        failed_models = []

        for model_name, weight_list in ALL_MDEs.items():
            for weights in weight_list:

                print(f"Running for model {model_name} with weights {weights}")
                model = get_mde_model(model_name, weights)
                try:
                    infer_depth(model, args)
                except Exception as e:
                    print(f"Model {model_name} with weights {weights} failed with error: {e}")
                    traceback.print_exc()

                    failed_models.append((model_name, weights))

        print(f'Inference failed for some cases: {failed_models}')
    else:
        model = get_mde_model(args.model_name, args.pretrained_weights)
        infer_depth(model, args)

if __name__ == '__main__':
    args = parse_args()
    run(args)