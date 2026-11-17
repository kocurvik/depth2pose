import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
import traceback

from tqdm import tqdm

import torch

torch.backends.cudnn.benchmark = True

from utils.system_info import save_metadata

ALL_MDEs = {
    # 'InfiniDepth': ['vitl'],
    'UniK3D': ['vitl'],
    'UniK3DCalib': ['vitl'],
    'Metric3DV2': ['vit_small', 'vit_large', 'vit_giant2'],
    'Metric3DV2Calib': ['vit_small', 'vit_large', 'vit_giant2'],
    'DepthAnythingV2': ['vits', 'vitb', 'vitl'],
    'DepthAnythingV3': ['DA3METRIC-LARGE', 'DA3MONO-LARGE'],
    'DepthAnythingV3Calib': ['DA3METRIC-LARGE', 'DA3MONO-LARGE'],
    'MoGeV1': ['moge-vitl'],
    'MoGeV2': ['moge-2-vitl'],
    'MoGeV1Calib': ['moge-vitl'],
    'MoGeV2Calib': ['moge-2-vitl'],
    'UniDepth2': ['vits14', 'vitb14', 'vitl14'],
    'UniDepth2Calib': ['vits14', 'vitb14', 'vitl14'],
    'UniDepth1': ['vitl14', 'cnvnxtl'],
    'UniDepth1Calib': ['vitl14', 'cnvnxtl'],
    'DepthPro': ['vitl'],
    'DepthProCalib': ['vitl'],
    'VGGT': ['VGGT-1B'],
    'Pi3': ['Pi3X'],
    'Pi3Calib': ['Pi3X'],
    'MapAnything': ['map-anything'],
    'MapAnythingCalib': ['map-anything']
    }




def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Inference script for depth estimation.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the depth estimation model.')
    parser.add_argument('--recalc', action='store_true', default=False, help='Whether inference requires intrinsics on input')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu).')
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained model weights.')
    parser.add_argument('--temp_out_path', default=None, type=str, help='Path to data directory.')
    parser.add_argument('out_path', type=str, help='Path to data directory.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset')

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)

def get_mde_model(model_name, weights):
    # if model_name == 'InfiniDepth':
    #     from depth_estimators.InfiniDepthWrapper import InfiniDepth
    #     return InfiniDepth(weights, requires_intrinsics=False)

    if model_name == '3DGS':
        from depth_estimators.NPZReader import NPZReader
        # weights is actually path here
        return NPZReader('gs_depths')

    if model_name in ('MoGeV1', 'MoGeV2', 'MoGeV1Calib', 'MoGeV2Calib'):
        from depth_estimators.MoGe import MoGe
        version = 2 if 'V2' in model_name else 1
        requires_intrinsics = 'Calib' in model_name
        return MoGe(weights, version=version, requires_intrinsics=requires_intrinsics)

    elif model_name in ('Metric3DV2', 'Metric3DV2Calib'):
        from depth_estimators.Metric3D import Metric3D
        return Metric3D(weights, version=2, requires_intrinsics='Calib' in model_name)

    elif model_name in ('UniK3D', 'UniK3DCalib'):
        from depth_estimators.UniK3D import UniK3D
        return UniK3D(weights, version=1, requires_intrinsics='Calib' in model_name)

    elif model_name in ('DepthPro', 'DepthProCalib'):
        from depth_estimators.DepthPro import DepthPro
        return DepthPro(weights, version=1, requires_intrinsics='Calib' in model_name)

    elif model_name in ('UniDepth1', 'UniDepth2', 'UniDepth1Calib', 'UniDepth2Calib'):
        from depth_estimators.UniDepth import UniDepth
        version = 2 if '2' in model_name else 1
        return UniDepth(weights, version=version, requires_intrinsics='Calib' in model_name)

    elif model_name in ('DepthAnythingV2', 'DepthAnythingV3', 'DepthAnythingV3Calib'):
        from depth_estimators.DepthAnything import DepthAnything
        version = 2 if 'V2' in model_name else 3
        return DepthAnything(weights, version=version, requires_intrinsics='Calib' in model_name)

    elif model_name == 'VGGT':
        from depth_estimators.VGGT import VGGT
        return VGGT(weights)

    elif model_name == 'Pi3' or model_name == 'Pi3Calib':
        from depth_estimators.Pi3 import Pi3
        return Pi3(weights, requires_intrinsics='Calib' in model_name)
    elif model_name == 'MapAnything' or model_name == 'MapAnythingCalib':
        from depth_estimators.MapAnything import MapAnything
        return MapAnything(weights, requires_intrinsics='Calib' in model_name)
    elif model_name == '3DGS':
        from depth_estimators.NPZReader import NPZReader
        return NPZReader()

    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


def get_depth_path(work_path, name, depth_name):
    return f'{os.path.join(work_path, name)}_depth_{depth_name}.h5'


def is_depth_complete(f_depth_path):
    """True if the depth file exists and has the completed tag. Prints the reason otherwise."""
    if not os.path.exists(f_depth_path):
        return False
    try:
        with h5py.File(f_depth_path, 'r') as f_check:
            if 'completed' in f_check:
                return True
            print(f"File {f_depth_path} exists but is incomplete.")
    except Exception:
        print(f"File {f_depth_path} exists but is broken.")
    return False


def read_image_list(work_path, name):
    with open(f'{os.path.join(work_path, name)}_image_list.txt', 'r') as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def infer_depth_scene(model, work_path, name, dataset_path, depth_name=None, pass_K=True, recalc=False,
                      temp_out_path=None, load_model=True):
    """Run the MDE on all images of one scene and store the results in <work_path>/<name>_depth_<depth_name>.h5.

    Args:
        model: a BaseDepthEstimator.
        work_path: directory with the extracted scene files (<name>.h5, <name>_image_list.txt).
        name: scene name.
        dataset_path: directory with the scene images (image names in the image list are relative to it).
        depth_name: name of the depth file, model.name if None.
        pass_K: pass the GT intrinsics to model.infer as K.
        recalc: recompute even if a completed depth file exists.
        temp_out_path: write the depth file into this directory instead of work_path.
        load_model: call model.load_model() before inference (set to False if the model is already loaded).

    Returns:
        True if inference was run, False if a completed depth file already existed.
    """
    if depth_name is None:
        depth_name = model.name
    name_path = os.path.join(work_path, name)

    f_depth_path = get_depth_path(work_path, name, depth_name)
    if not recalc and is_depth_complete(f_depth_path):
        print(f"File {f_depth_path} already exists and is completed. Skipping. Use --recalc arg to force recalculation.")
        return False

    if temp_out_path is not None:
        f_depth_path = get_depth_path(temp_out_path, name, depth_name)
        print(f"Using temp file {f_depth_path}")

    if load_model:
        print("Loading Model")
        model.load_model()

    # an incomplete file from an interrupted run is overwritten
    print(f"Creating {f_depth_path}")
    f_depth = h5py.File(f_depth_path, 'w')
    save_metadata(f_depth)

    image_list = read_image_list(work_path, name)

    f_image_dict = {}
    with h5py.File(f'{name_path}.h5', 'r') as f_images:
        for image_name in image_list:
            f_image_dict[f'{image_name}_K'] = np.array(f_images[f'{image_name}_K'])
            f_image_dict[f'{image_name}_size'] = np.array(f_images[f'{image_name}_size'])
            f_image_dict[f'{image_name}_size_orig'] = np.array(f_images[f'{image_name}_size_orig'])

    # compressing and writing the results runs in a background thread while the next image is processed
    # (h5py releases the GIL while writing), at most max_pending results are waiting to be written
    max_pending = 2
    pending = deque()
    with ThreadPoolExecutor(max_workers=1) as writer:
        try:
            for image_name in tqdm(image_list):
                K = f_image_dict[f'{image_name}_K']
                img_path = os.path.join(dataset_path, Path(PureWindowsPath(image_name)))
                size = f_image_dict[f'{image_name}_size']
                size_orig = f_image_dict[f'{image_name}_size_orig']
                kwargs = {'K': K} if pass_K else {}
                if (size != size_orig).any():
                    # if we resized during dataset generation we need to resize here
                    out_dict = model.infer(img_path, size=size, **kwargs)
                else:
                    out_dict = model.infer(img_path, **kwargs)

                pending.append(writer.submit(save_depth_result, f_depth, image_name, out_dict))
                while len(pending) > max_pending:
                    pending.popleft().result()  # re-raises errors from the writer
            while pending:
                pending.popleft().result()
        except BaseException:
            # also on errors / Ctrl+C: finish the already submitted writes and close the file (it has no completed
            # tag, so it is recomputed by the next run)
            for future in pending:
                future.exception()
            f_depth.close()
            raise

    f_depth.create_group('completed')
    print(f"{f_depth_path} completed")
    f_depth.close()
    return True


def save_depth_result(f_depth, image_name, out_dict):
    """Write the output of model.infer for one image into the depth file."""
    depth_f32 = out_dict['depth']
    save_as_f32 = False
    finite_mask = np.isfinite(depth_f32)
    if finite_mask.any():
        finite_vals = depth_f32[finite_mask]
        extremes_f32 = np.array([finite_vals.min(), finite_vals.max()], dtype=np.float32)
        extremes_f16 = extremes_f32.astype(np.float16)
        if not np.all(np.isfinite(extremes_f16)):
            print(f"WARNING: {image_name} depth float32->float16 overflow: min={extremes_f32[0]:.4f}, max={extremes_f32[1]:.4f}. Saving as float32.")
            save_as_f32 = True
        else:
            rel_err = np.max(np.abs(extremes_f32.astype(np.float64) - extremes_f16.astype(np.float64)) / np.abs(extremes_f32.astype(np.float64)))
            if rel_err > 1e-3:
                print(f"WARNING: {image_name} depth float32->float16 max relative error at extremes: {rel_err:.2e} (min={extremes_f32[0]:.4f}, max={extremes_f32[1]:.4f}). Saving as float32.")
                save_as_f32 = True
    depth_save = depth_f32 if save_as_f32 else depth_f32.astype(np.float16)

    f_depth.create_dataset(f'{image_name}_depth', data=depth_save, compression='gzip', chunks=True)
    f_depth.create_dataset(f'{image_name}_runtime', data=out_dict['runtime'])

    if 'inference_K' in out_dict.keys():
        f_depth.create_dataset(f'{image_name}_inference_K', data=out_dict['inference_K'], compression='gzip', chunks=True)
    if 'K' in out_dict.keys():
        f_depth.create_dataset(f'{image_name}_K', data=out_dict['K'], compression='gzip', chunks=True)


def infer_depth(model, args):
    """Command line wrapper of infer_depth_scene (GT intrinsics are always passed, the model decides whether to
    use them via requires_intrinsics)."""
    return infer_depth_scene(model, args.out_path, args.name, args.dataset_path, recalc=args.recalc,
                             temp_out_path=args.temp_out_path)


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
