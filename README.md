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

  [![arXiv](https://img.shields.io/badge/arXiv-tba.tba-b31b1b.svg)](https://arxiv.org/abs/tba)
  [![Project Page](https://img.shields.io/static/v1?label=Project&message=Website&color=red)](https://kocurvik.github.io/depth2pose)
  [![Project Page](https://img.shields.io/static/v1?label=D2P&message=Dataset(Sample)&color=blue)](https://huggingface.co/datasets/floodgab/d2p_dataset_example)
  [![Project Page](https://img.shields.io/static/v1?label=D2P&message=Dataset&color=blue)](https://huggingface.co/datasets/floodgab/d2p_dataset)
  
  </div>
</p>

# About
This repository contains the full evalautation code the depth2pose framework for estimting monocular depth estimators.

## Dataset download

You can download the full D2P dataset used in the paper from [hugging face](https://huggingface.co/datasets/floodgab/d2p_dataset)

## Installation

See [INSTALL.md](INSTALL.md).

## Evaluation

### Running Full Evaluation

To run the full evaluation for all MDEs we provide code to create array jobs for a SLURM cluster. If a cluster is not available the scripts should default to local execution.

Modify the file `datasets/d2p_benchmark.json` and `datasets/standard_benchmark.json` with paths to the datasets.

Then you can run the following:
```
# extract pairs
python datasets/colmap.py --config_path datasets/d2p_benchmark.json
python datasets/colmap.py --config_path datasets/standard_benchmark.json

# match pairs (use matchers/splg.py to obtain SP+LG matches in the same way)
python matchers/loma_matching.py --config_path datasets/d2p_benchmark.json
python matchers/loma_matching.py --config_path datasets/standard_benchmark.json

# infer depth
python spawn_infer_depth.py  --config_path datasets/d2p_benchmark.json
python spawn_infer_depth.py  --config_path datasets/standard_benchmark.json

# evaluate depth for the standard benchmark
python eval_depth.py  --config_path datasets/d2p_benchmark.json

# eval pose estimation accuracy
python spawn_eval_pose.py  --config_path datasets/d2p_benchmark.json --matches loma_2048_noresize
python spawn_eval_pose.py  --config_path datasets/standard_benchmark.json --matches loma_2048_noresize

# finally extract results
python utils/extract_results.py  --config_path datasets/d2p_benchmark.json --matches loma_2048_noresize
python utils/extract_results.py  --config_path datasets/standard_benchmark.json --matches loma_2048_noresize
```
Except for `colmap.py` you will need to modify the args to select the correct partition and set time limits.

### Adding a new MDE

To add a new MDE you need to add the required code (into depth_estimators folder) implementing the base class in `depth_estimators/base.py`. Then you also have to modify ``depth_estimators/infer_depth.py` to include the MDE along with its variants similar to the other methods.

To run the mde you need the perform all the steps. But when running:
```
python spawn_infer_depth.py  --config_path datasets/d2p_benchmark.json
python spawn_infer_depth.py  --config_path datasets/standard_benchmark.json
```
You add the following arg based on the name of the depth in ALL_MDEs_DICT:
```
--model_name your_model_name
```
This will run only the new model. You may run pose estimation with depth provided for only one model and then export the results with the same command as for the full evaluation.
