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
This repository contains the full evaluation code of the Depth2Pose framework for evaluating monocular depth estimators. Our [paper](https://arxiv.org/abs/2605.19797) was accepted in the E&D track of NeurIPS 2026.

*TLDR:* Using standard metrics to evaluate MDEs requires known ground truth depth, which may be hard to acquire. In the Depth2Pose framework we evaluate the depth via two-view relative pose accuracy using the [RePoseD framework](https://kocurvik.github.io/reposed/). This approach only requires a collection of images from which a COLMAP reconstruction can be obtained. By design, depth2pose focuses its evaluation on the quality of geometrically relevant parts of the images and thus captures the usefulness of the estimated depth for downstream geometric tasks.

There are two ways to use this repository:

- **[Evaluating your MDE on D2P](#evaluating-your-mde-on-d2p)**: a single script, `eval_my_mde.py`, that runs on one machine with a GPU. It downloads the D2P dataset, runs your MDE, estimates the relative poses and adds your results to the results table. This is the recommended way to evaluate a new MDE.
- **[Running the full benchmark](#running-the-full-benchmark)**: the complete pipeline used for the paper (pair extraction, matching, depth inference and pose estimation for all MDEs and datasets), designed for SLURM clusters.

If you find this work useful please consider citing:
```
@inproceedings{kocur2026depth2pose,
  title={Depth2Pose: A Pose-Based Benchmark for Monocular Depth Estimation without Ground-Truth Depth},
  author={Kocur, Viktor and Aung, Sithu and Flood, Gabrielle and Ding, Yaqing and Bujnak, Lukas and Sattler, Torsten and Kukelova, Zuzana},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

# Evaluating your MDE on D2P

## Installation

`eval_my_mde.py` needs only a few packages besides the dependencies of your MDE:
```
conda create -n d2p python=3.10
conda activate d2p
# install PyTorch (https://pytorch.org) and the dependencies of your MDE, then:
pip install numpy h5py opencv-python pandas scipy matplotlib prettytable psutil GitPython tqdm huggingface_hub
pip install git+https://github.com/kocurvik/PoseLib.git@mdrpbench
```
The relative pose solvers using monocular depth (RePoseD) are in our [fork of PoseLib](https://github.com/kocurvik/PoseLib/tree/mdrpbench), which is compiled from source during the installation. Other PoseLib versions do not provide them; the script checks this at the start and prints installation instructions if the installed PoseLib is not compatible.

## Implementing your MDE

Implement the `MyMDE` class at the top of `eval_my_mde.py`:

| method | what it does |
|---|---|
| `load_model()` | Loads the network. Called once before the inference. |
| `name` | Name of your MDE in the results table, e.g. `MyMDE` or `MyMDE-vitl` for a specific variant. When run with `--use_gt_K`, `Calib` is added (`MyMDECalib-vitl`). |
| `infer(image, K=None, size=None)` | Predicts depth for the image at path `image`. `K` is the ground-truth 3x3 intrinsic matrix (in pixels), passed only with `--use_gt_K`. Returns a dict with `depth` (HxW depth map at the full image resolution, metric or up to scale), `runtime` (inference time in nanoseconds) and optionally `K` (3x3 intrinsics predicted by the MDE, in pixels). |

Complete examples are in [`example_scripts/`](example_scripts): [`eval_moge_v2.py`](example_scripts/eval_moge_v2.py) for MoGe-2, [`eval_moge_v3.py`](example_scripts/eval_moge_v3.py) for MoGe-3 and [`eval_surge.py`](example_scripts/eval_surge.py) for [SurGe](https://github.com/karimknaebel/surge). They can be run in the same way as `eval_my_mde.py`.

## Running the evaluation

Run from the root of this repository:
```
# the MDE predicts depth (and optionally intrinsics) from the image only
python eval_my_mde.py
# the MDE gets the ground-truth intrinsics as input
python eval_my_mde.py --use_gt_K
```

The script:
1. downloads D2P from [hugging face](https://huggingface.co/datasets/floodgab/d2p_dataset) into `./d2p` if it is not there yet (about 8 GB),
2. runs your MDE on all 1953 benchmark images,
3. estimates the relative poses of the 250 image pairs of each of the 24 scenes with 1000 RANSAC iterations using the solvers
   - `calib`: ground-truth intrinsics, minimal solver and scoring using the depth,
   - `calib_ro`: as `calib`, but RANSAC scores the models only by the reprojection error,
   - `mdecalib`, `mdecalib_ro`: as above with the intrinsics predicted by your MDE (only if `infer` returns `K` and `--use_gt_K` is not set),
4. prints the pose mAA(10) for the statues and vegetation scenes and appends the results to the results table (see [below](#submitting-your-results)).

| option | default | |
|---|---|---|
| `--use_gt_K` | off | pass the ground-truth intrinsics to `infer` |
| `--matches` | `loma` | matches used for pose estimation: `loma` (LoMa) or `splg` (SuperPoint + LightGlue) |
| `--num_workers` | 8 | number of processes for pose estimation |
| `--data_path` | `./d2p` | where the dataset is downloaded to; depth maps and per scene results are also stored there |
| `--recalc` | off | recompute depth and poses from a previous run, use after changing your MDE |
| `--overwrite` | off | replace the results of an MDE with the same name already in the results table |
| `--out_dir` | `csv_results` | directory with the results tables |

Depth maps and pose results are stored per scene in `--data_path`. If the script is interrupted, running it again continues with the first scene that was not finished. For reference, evaluating MoGe v2 on a laptop (RTX A4000 Laptop GPU, 12 workers) took about 35 minutes for the depth inference and 18 minutes for the pose estimation.

## Submitting your results

The results are appended to `csv_results/d2p_<matches>_slim_pose_results.csv` (per scene) and `csv_results/per_group_d2p_<matches>_slim_pose_results.csv` (mean over the statues and vegetation scenes), which contain the results of all MDEs evaluated in the paper. To add your MDE to the results table, open a pull request with the updated CSV files and a link to your MDE. See [#2](https://github.com/kocurvik/depth2pose/pull/2) (SurGe) for an example of such a pull request.

If an MDE with the same name already is in the table, the script stops before running anything. Choose a different name, use `--overwrite` to replace its results (e.g. after fixing a bug in your MDE), or use `--out_dir` to write the results elsewhere.

# Running the full benchmark

## Installation

See [INSTALL.md](INSTALL.md).

## Dataset download

The D2P dataset used in the paper is available on [hugging face](https://huggingface.co/datasets/floodgab/d2p_dataset). It contains two folders:

- `d2p_dataset/{statues,vegetation}/<scene>/`: undistorted images, COLMAP reconstructions and the lists of images and image pairs used in the benchmark.
- `d2p_extracted/{statues,vegetation}/<scene>/`: the pre-extracted ground-truth poses and intrinsics (`<scene>.h5`) and the LoMa (`<scene>_loma_2048_noresize.h5`) and SuperPoint+LightGlue (`<scene>_splg_2048_noresize.h5`) matches for all evaluated pairs, in the format used by this code. With these files the pair extraction and matching steps can be skipped.

The format of all files is described in the [dataset card](https://huggingface.co/datasets/floodgab/d2p_dataset). To download the files needed for evaluation into `./d2p` (the location expected by `datasets/d2p_hf.json`), run from the root of this repository:
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('floodgab/d2p_dataset', repo_type='dataset', local_dir='d2p', allow_patterns=['d2p_dataset/*', 'd2p_extracted/*'])"
```

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

To evaluate a new MDE on D2P only, use `eval_my_mde.py` as described [above](#evaluating-your-mde-on-d2p). To add it to the full evaluation pipeline you need to add the required code (into depth_estimators folder) implementing the base class in `depth_estimators/base.py`. Then you also have to modify `depth_estimators/infer_depth.py` to include the MDE along with its variants (in `ALL_MDEs` and `get_mde_model`) similar to the other methods.

To evaluate the MDE run the same steps as above (for D2P only the steps in [Evaluating on D2P using the pre-extracted data](#evaluating-on-d2p-using-the-pre-extracted-data) are needed). But when running:
```
python spawn_infer_depth.py --config_path datasets/d2p_hf.json
```
You add the following arg based on the name of the MDE in `ALL_MDEs`:
```
--model_name your_model_name
```
This will run only the new model. You may run pose estimation with depth provided for only one model and then export the results with the same command as for the full evaluation. 
