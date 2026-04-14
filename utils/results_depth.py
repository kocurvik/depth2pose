import json
import numpy as np
from pathlib import Path
from prettytable import PrettyTable



def print_results_focal(metrics):
    tab = PrettyTable(['solver', 'median pose err', 'median f err',
                       'pose mAA(10)', 'f mAA(0.1)', 'mean mde runtime', 'mean pose runtime', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp_name, m in metrics.items():
        tab.add_row([exp_name, m['median_pose_err'], m['median_f_err'],
                     m['pose_mAA_10'], m['f_mAA_10'], m['mean_mde_runtime'], m['mean_runtime'], m['mean_inliers']])
    print(tab)



def main(root, depth_models: list[str], datasets: list[str]):
    metrics = ["A.Rel ↓", "delta_1 ↑"] * len(datasets)
    headers = []
    headers.extend([datasets[0], "-"])
    headers.extend([datasets[1], "--"])
    metric_table = PrettyTable(["MDE", *headers])
    metric_table.align['MDE'] = 'l'
    metric_table.float_format = '0.2'
    metric_table.add_row(["", *metrics])

    scale_table = PrettyTable(["MDE", *headers])
    scale_table.align['MDE'] = 'l'
    scale_table.float_format = '0.2'
    scale_table.add_row(["", *metrics])

    affine_table = PrettyTable(["MDE", *headers])
    affine_table.align['MDE'] = 'l'
    affine_table.float_format = '0.2'
    affine_table.add_row(["", *metrics])

    for depth_model in depth_models:
        metric_result_path = root / depth_model / "metric_depth_results.json"
        scale_result_path = root / depth_model / "scale_inv_depth_results.json"
        affine_result_path = root / depth_model / "affine_inv_depth_results.json"

        with open(metric_result_path) as f:
            metric_results = json.load(f)
        with open(scale_result_path) as f:
            scale_results = json.load(f)
        with open(affine_result_path) as f:
            affine_results = json.load(f)

        metric_table.add_row([depth_model,
            metric_results[datasets[0]]["A.Rel"],
            metric_results[datasets[0]]["d1"],
            metric_results[datasets[1]]["A.Rel"],
            metric_results[datasets[1]]["d1"]
        ])

        scale_table.add_row([depth_model,
            scale_results[datasets[0]]["A.Rel_si"],
            scale_results[datasets[0]]["d1_si"],
            scale_results[datasets[1]]["A.Rel_si"],
            scale_results[datasets[1]]["d1_si"]
        ])

        affine_table.add_row([depth_model,
            affine_results[datasets[0]]["A.Rel_ssi"],
            affine_results[datasets[0]]["d1_ssi"],
            affine_results[datasets[1]]["A.Rel_ssi"],
            affine_results[datasets[1]]["d1_ssi"]
        ])
    print("Metric Depth Results")
    print(metric_table)
    print("Scale-invariant Depth Results")
    print(scale_table)
    print("Affine-invariant Depth Results")
    print(affine_table)



if __name__ == '__main__':
    root = Path("/mnt/data/gg/benchmarks_results")
    depth_models = sorted(list(root.glob("*")))
    depth_models = [depth_model.stem for depth_model in depth_models if depth_model.is_dir()]

    datasets = ['eth3d', 'mean']
    main(root, depth_models, datasets)
