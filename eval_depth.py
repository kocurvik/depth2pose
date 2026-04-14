import json
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "./")
from datasets.dataloader import EvalDataLoaderPipeline
from utils.metrics import DepthMetrics, key_average


def get_depth_from_h5(dataset_name, dataset_config, depth_model_name, scene_name, file_name):
    # all_depths = defaultdict(dict)
    with h5py.File(Path(dataset_config['depth']) / f'{dataset_name}_depth_{depth_model_name}.h5', 'r') as f_depth_h5:
        # for file in f_depth_h5.keys():
        #     scene_name, _, depth_file_name = file.split("\\")
        #     file_name = depth_file_name.split(".")[0]
        #     all_depths[scene_name][file_name] = np.array(f_depth_h5[depth_file_name]).astype(np.float32)
        depth_key_name = f"{scene_name}\\images\\{file_name}_depth"
        depth = np.array(f_depth_h5[depth_key_name])
    depth[depth <= 0] = np.inf
    return depth

def evaluate_model(mde_model, dataset_config, save_dir, device):
    metric_result_path = save_dir / "metric_depth_results.json"
    scale_result_path = save_dir / "scale_inv_depth_results.json"
    affine_result_path = save_dir / "affine_inv_depth_results.json"

    metric_fn = DepthMetrics()
    all_metric_depth_results, all_scale_depth_results, all_affine_depth_results = {}, {}, {}

    # iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(
        list(dataset_config.items()), desc="Benchmarks"
    ):
        metric_result_list, scale_result_list, affine_result_list = [], [], []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar,
        ):
            # iterate over the samples in the dataset
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

                pd_depth = get_depth_from_h5(benchmark_name, benchmark_config, mde_model, scenename, filename)
                pd_depth = torch.from_numpy(pd_depth).to(torch.float32).to(device)

                metric_results = metric_fn.compute_metric_depth(gt_depth, pd_depth, depth_mask)
                scale_results = metric_fn.compute_scale_inv_depth(gt_depth, pd_depth, depth_mask)
                affine_results = metric_fn.compute_affine_inv_depth(gt_depth, pd_depth, depth_mask)
                metric_result_list.append(metric_results)
                scale_result_list.append(scale_results)
                affine_result_list.append(affine_results)

                # if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                #     Path(metric_result_path).write_text(
                #         json.dumps(
                #             {**all_metric_depth_results, benchmark_name: key_average(metric_result_list)},
                #             indent=4,
                #         )
                #     )
                #     Path(scale_result_path).write_text(
                #         json.dumps(
                #             {**all_scale_depth_results, benchmark_name: key_average(scale_result_list)},
                #             indent=4,
                #         )
                #     )
                #     Path(affine_result_path).write_text(
                #         json.dumps(
                #             {**all_affine_depth_results, benchmark_name: key_average(affine_result_list)},
                #             indent=4,
                #         )
                #     )
                pbar.update(1)

        all_metric_depth_results[benchmark_name] = key_average(metric_result_list)
        all_scale_depth_results[benchmark_name] = key_average(scale_result_list)
        all_affine_depth_results[benchmark_name] = key_average(affine_result_list)

    all_metric_depth_results["mean"] = key_average(list(all_metric_depth_results.values()))
    all_scale_depth_results["mean"] = key_average(list(all_scale_depth_results.values()))
    all_affine_depth_results["mean"] = key_average(list(all_affine_depth_results.values()))
    Path(metric_result_path).write_text(json.dumps(all_metric_depth_results, indent=4))
    Path(scale_result_path).write_text(json.dumps(all_scale_depth_results, indent=4))
    Path(affine_result_path).write_text(json.dumps(all_affine_depth_results, indent=4))


def get_depth_models(root: str):
    root = Path(root)
    dataset_dir = [dataset.stem for dataset in root.glob("*") if dataset.is_dir()][0]
    depth_models = sorted(list((root / dataset_dir).glob("*.h5")))
    depth_models = [depth_model.stem.split("_depth_")[-1] for depth_model in depth_models]
    return depth_models

def main():
    config_path = "./datasets/all_benchmarks.json"
    with open(config_path) as f:
        dataset_config = json.load(f)

    device = torch.device("cuda")
    depth_models_path = "/mnt/data/gg/benchmarks_depth"
    save_results_path = "/mnt/data/gg/benchmarks_results"

    depth_models = get_depth_models(depth_models_path)
    for model_name in depth_models:
        print(model_name)
        # if model_name != "Metric3DV2-vit_giant2": continue
        save_dir = Path(save_results_path) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        evaluate_model(model_name, dataset_config, save_dir, device)


if __name__ == "__main__":
    main()
