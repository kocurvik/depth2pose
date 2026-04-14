import argparse
import json
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from utils.results import get_mde_list

sys.path.insert(0, "./")
from datasets.dataloader import EvalDataLoaderPipeline
from utils.metrics import DepthMetrics, key_average


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('all_results_path')
    parser.add_argument('config_path')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu to be used for extraction')
    parser.add_argument('--recalc', action='store_true', default = False,
                        help='whether to recalculate even previously calculated depths')
    return parser.parse_args()


def get_depth_from_h5(f_depth_h5, scene_name, file_name):
    depth_key_name = f"{scene_name}\\images\\{file_name}_depth"
    depth = np.array(f_depth_h5[depth_key_name])
    depth[depth <= 0] = np.inf
    return depth

def evaluate_model(mde_model, dataset_config, save_dir_all, device, recalc=False):
    metric_result_path = save_dir_all / "metric_depth_results.json"
    scale_result_path = save_dir_all / "scale_inv_depth_results.json"
    affine_result_path = save_dir_all / "affine_inv_depth_results.json"

    metric_fn = DepthMetrics()
    all_metric_depth_results, all_scale_depth_results, all_affine_depth_results = {}, {}, {}

    # iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(
        list(dataset_config.items()), desc="Benchmarks"
    ):
        single_results_path = Path(benchmark_config['depth']) / 'depth_results' / f'{mde_model}.json'

        if os.path.exists(single_results_path) and not recalc:
            continue

        metric_result_list, scale_result_list, affine_result_list = [], [], []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar,
            h5py.File(Path(benchmark_config['depth']) / f'{benchmark_name}_depth_{mde_model}.h5',
                      'r') as f_depth_h5
        ):
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                }
                scenename, filename = sample["scenename"], sample["filename"]
                _, gt_depth, depth_mask = (
                    sample["image"],
                    sample["depth"],
                    sample["depth_mask"]
                )
                # pd_depth_data = np.load(eval_data_pipe.path / scenename / mde_model / f"{filename}.npz")
                # pd_depth, pd_intrinsic = pd_depth_data["depth"], pd_depth_data["K"]

                pd_depth = get_depth_from_h5(f_depth_h5, scenename, filename)
                pd_depth = torch.from_numpy(pd_depth).to(torch.float32).to(device)

                metric_results = metric_fn.compute_metric_depth(gt_depth, pd_depth, depth_mask)
                scale_results = metric_fn.compute_scale_inv_depth(gt_depth, pd_depth, depth_mask)
                affine_results = metric_fn.compute_affine_inv_depth(gt_depth, pd_depth, depth_mask)
                metric_result_list.append(metric_results)
                scale_result_list.append(scale_results)
                affine_result_list.append(affine_results)

                pbar.update(1)

        single_results = {'metric': key_average(metric_result_list), 'scale': key_average(scale_result_list),
                          'affine': key_average(affine_result_list)}

        Path(single_results_path).parent.mkdir(exist_ok=True, parents=True)
        Path(single_results_path).write_text(json.dumps(single_results, indent=4))

        all_metric_depth_results[benchmark_name] = key_average(metric_result_list)
        all_scale_depth_results[benchmark_name] = key_average(scale_result_list)
        all_affine_depth_results[benchmark_name] = key_average(affine_result_list)

    all_metric_depth_results["mean"] = key_average(list(all_metric_depth_results.values()))
    all_scale_depth_results["mean"] = key_average(list(all_scale_depth_results.values()))
    all_affine_depth_results["mean"] = key_average(list(all_affine_depth_results.values()))

    Path(metric_result_path).write_text(json.dumps(all_metric_depth_results, indent=4))
    Path(scale_result_path).write_text(json.dumps(all_scale_depth_results, indent=4))
    Path(affine_result_path).write_text(json.dumps(all_affine_depth_results, indent=4))


def get_depth_models(processed_dataset_path):
    depth_models = sorted(list((Path(processed_dataset_path)).glob("*.h5")))
    depth_models = [depth_model.stem.split("_depth_")[-1] for depth_model in depth_models]
    return depth_models

def main():
    args = parse_args()
    config_path = args.config_path

    with open(config_path) as f:
        dataset_config = json.load(f)

    device = torch.device(args.device)

    first_dataset_name = list(dataset_config.keys())[0]
    depth_models = get_mde_list(first_dataset_name, dataset_config[first_dataset_name]['depth'])

    for model_name in depth_models:
        print(model_name)
        # if model_name != "Metric3DV2-vit_giant2": continue
        save_dir = Path(args.all_results_path) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        evaluate_model(model_name, dataset_config, save_dir, device, recalc=args.recalc)


if __name__ == "__main__":
    main()
