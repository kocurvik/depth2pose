import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "./")
from datasets.dataloader import EvalDataLoaderPipeline
from utils.metrics import DepthMetrics, key_average


def main():
    config_path = "./datasets/all_benchmarks.json"
    mde_model = "MoGeV2-moge-2-vitl"
    result_path = "depth_results.json"

    device = torch.device("cuda")
    metric_fn = DepthMetrics()

    with open(config_path) as f:
        config = json.load(f)

    all_metrics = {}
    # iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(
        list(config.items()), desc="Benchmarks"
    ):
        metrics_list = []
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
                image, gt_depth, depth_mask, gt_intrinsic = (
                    sample["image"],
                    sample["depth"],
                    sample["depth_mask"],
                    sample["intrinsics"],
                )
                pd_depth_data = np.load(
                    eval_data_pipe.path / scenename / mde_model / f"{filename}.npz"
                )
                pd_depth, pd_intrinsic = pd_depth_data["depth"], pd_depth_data["K"]
                pd_depth = torch.from_numpy(pd_depth).to(torch.float32).to(device)
                results = metric_fn.compute_affine_inv_depth(
                    gt_depth, pd_depth, depth_mask
                )
                metrics_list.append(results)

                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    Path(result_path).write_text(
                        json.dumps(
                            {**all_metrics, benchmark_name: key_average(metrics_list)},
                            indent=4,
                        )
                    )
                pbar.update(1)

        all_metrics[benchmark_name] = key_average(metrics_list)

    all_metrics["mean"] = key_average(list(all_metrics.values()))
    Path(result_path).write_text(json.dumps(all_metrics, indent=4))


if __name__ == "__main__":
    main()
