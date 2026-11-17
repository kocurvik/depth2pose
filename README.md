<p align="center">
  <h1 align="center">Depth2Pose: A Pose-Based Benchmark for Monocular Depth Estimation without Ground-Truth Depth</h1>
  <p align="center">
    <span class="author-block">
      Viktor Kocur
      ·
      Sithu Aung
      ·
      Gabrielle Flood
      ·
      Yaqing Ding
      ·
      Lukáš Bujňák
      ·
      Torsten Sattler
      ·
      Zuzana Kukelova
  </p>
  <div align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-2605.19797-b31b1b.svg)](https://arxiv.org/abs/2605.19797)
  [![Project Page](https://img.shields.io/static/v1?label=Project&message=Website&color=red)](https://kocurvik.github.io/depth2pose)
  [![Project Page](https://img.shields.io/static/v1?label=D2P&message=Dataset(Sample)&color=blue)](https://huggingface.co/datasets/floodgab/d2p_dataset_example)
  [![Project Page](https://img.shields.io/static/v1?label=D2P&message=Dataset&color=blue)](https://huggingface.co/datasets/floodgab/d2p_dataset)
  
  </div>
</p>

# About
This repository contains the full evalautation code the depth2pose framework for estimting monocular depth estimators. Our [paper](https://arxiv.org/abs/2605.19797) was accepted in the E&D track of NeurIPS 2026.

*TLDR:* Using standard metrics to evaluate MDEs requires known ground truth depth, which may be hard to acquire. In the Depth2Pose framework we evaluate the depth via two-view relative pose accuracy using the [RePoseD framework](https://kocurvik.github.io/reposed/). This approach only requires a collection of images from which a COLMAP reconstruction can be obtained. By design, depth2pose focuses its evaluation on the quality of geometrically relevant parts of the images and thus captures the usefulness of the estimated depth for downstream geometric tasks.

## Dataset download

The D2P dataset used in the paper is available on [hugging face](https://huggingface.co/datasets/floodgab/d2p_dataset). It contains two folders:

- `d2p_dataset/{statues,vegetation}/<scene>/`: undistorted images, COLMAP reconstructions and the lists of images and image pairs used in the benchmark.
- `d2p_extracted/{statues,vegetation}/<scene>/`: the pre-extracted ground-truth poses and intrinsics (`<scene>.h5`) and the LoMa (`<scene>_loma_2048_noresize.h5`) and SuperPoint+LightGlue (`<scene>_splg_2048_noresize.h5`) matches for all evaluated pairs, in the format used by this code. With these files the pair extraction and matching steps can be skipped.

The format of all files is described in the [dataset card](https://huggingface.co/datasets/floodgab/d2p_dataset). To download the files needed for evaluation into `./d2p` (the location expected by `datasets/d2p_hf.json`), run from the root of this repository:
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('floodgab/d2p_dataset', repo_type='dataset', local_dir='d2p', allow_patterns=['d2p_dataset/*', 'd2p_extracted/*'])"
```

## Installation

See [INSTALL.md](INSTALL.md).

## Evaluation

All steps below create array jobs for a SLURM cluster using [submitit](https://github.com/facebookincubator/submitit). If a cluster is not available, the scripts fall back to local execution. When running on a cluster, set `--account`, `--queue`, `--timeout_min` and `--mem_gb` of the `spawn_*.py`, `matchers/*.py` and `eval_depth.py` scripts for your cluster.

The datasets are defined by JSON configs in `datasets/`. For each dataset, `path` points to the images and COLMAP models and `work_path` to the directory where the extracted pairs, matches, depth maps and results are stored (one subfolder per scene):

- `datasets/d2p_hf.json`: D2P downloaded from hugging face as described above. Paths are relative to the root of this repository; change them to absolute paths if you store the data elsewhere.
- `datasets/d2p_benchmark.json`, `datasets/standard_benchmark.json`: the configs we used on our cluster; modify the paths if you want to use them.

### Evaluating on D2P using the pre-extracted data

After downloading the dataset, the ground truth and matches are already in place, so only depth inference and pose estimation need to be run:
```
# infer depth for all MDEs
python spawn_infer_depth.py --config_path datasets/d2p_hf.json

# eval pose estimation accuracy (use --matches splg_2048_noresize for SP+LG matches)
python spawn_eval_pose_jobs.py --config_path datasets/d2p_hf.json --matches loma_2048_noresize

# eval methods that directly regress depth and relative pose
python spawn_eval_regressed_pose_jobs.py --config_path datasets/d2p_hf.json

# extract results into csv_results/ (add -r for the regressed poses)
python extract_results.py --config_path datasets/d2p_hf.json --matches loma_2048_noresize
```

### Running Full Evaluation

To run the full pipeline including pair extraction and matching (e.g. for the standard benchmark or for a new dataset), modify the paths in `datasets/d2p_benchmark.json` and `datasets/standard_benchmark.json` and run the following:
```
# extract pairs
python datasets/colmap.py --config_path datasets/d2p_benchmark.json
python datasets/colmap.py --config_path datasets/standard_benchmark.json

# match pairs (use matchers/splg.py to obtain SP+LG matches in the same way)
python matchers/loma_matching.py --config_path datasets/d2p_benchmark.json
python matchers/loma_matching.py --config_path datasets/standard_benchmark.json

# infer depth
python spawn_infer_depth.py --config_path datasets/d2p_benchmark.json
python spawn_infer_depth.py --config_path datasets/standard_benchmark.json

# evaluate depth for the standard benchmark (D2P has no ground-truth depth)
python eval_depth.py datasets/standard_benchmark.json

# eval pose estimation accuracy
python spawn_eval_pose_jobs.py --config_path datasets/d2p_benchmark.json --matches loma_2048_noresize
python spawn_eval_pose_jobs.py --config_path datasets/standard_benchmark.json --matches loma_2048_noresize

# eval methods that directly regress depth and relative pose
python spawn_eval_regressed_pose_jobs.py --config_path datasets/d2p_benchmark.json
python spawn_eval_regressed_pose_jobs.py --config_path datasets/standard_benchmark.json

# finally extract results (add -r for the regressed poses)
python extract_results.py --config_path datasets/d2p_benchmark.json --matches loma_2048_noresize
python extract_results.py --config_path datasets/standard_benchmark.json --matches loma_2048_noresize
```
If a scene folder in `path` contains `<scene>_image_list.txt` and `<scene>_image_pairs.txt` (as in the D2P download), `colmap.py` uses these lists instead of sampling new pairs.

### Adding a new MDE

To add a new MDE you need to add the required code (into depth_estimators folder) implementing the base class in `depth_estimators/base.py`. Then you also have to modify `depth_estimators/infer_depth.py` to include the MDE along with its variants (in `ALL_MDEs` and `get_mde_model`) similar to the other methods.

To evaluate the MDE run the same steps as above (for D2P only the steps in [Evaluating on D2P using the pre-extracted data](#evaluating-on-d2p-using-the-pre-extracted-data) are needed). But when running:
```
python spawn_infer_depth.py --config_path datasets/d2p_hf.json
```
You add the following arg based on the name of the MDE in `ALL_MDEs`:
```
--model_name your_model_name
```
This will run only the new model. You may run pose estimation with depth provided for only one model and then export the results with the same command as for the full evaluation. 
