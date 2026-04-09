# Benchmarking MDE models

## Installation

Install from yml and requirements file:
```bash
conda env create --file environment.yml
conda activate mdrpbench
conda install --file requirements.txt
```

Or you can create your own from below:
Base torch environment:
```bash
conda create -n mdrpbench python=3.10
conda activate mdrpbench
conda install nvidia::cuda-toolkit==12.8.0
pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128
```

Install MDE models:
```bash
# install depth-anything-3
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# install depth-anything-2
pip install git+https://github.com/badayvedat/Depth-Anything-V2.git@2ca8cd3dd7dc8b0a7386126ad65e58c7fc44e925

# install depth-pro
pip install git+https://github.com/apple/ml-depth-pro.git

# install moge
pip install git+https://github.com/microsoft/MoGe.git

# install unidepth
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git

# install unik3d
pip install git+https://github.com/kocurvik/UniK3D.git
```

Install matching models:
```bash
# install romav1
pip install git+https://github.com/Parskatt/RoMa.git

# install lightglue
pip install git+https://github.com/cvg/LightGlue.git
```

## Depth Evaluation

### Extract Mono Depths

Extract depth with a single MDE model:
```bash
# example - MoGeV2
python depth_estimators/gen_depth.py \
    --model_name MoGeV2 \
    --name ETH3D \
    --pretrained_weights moge-2-vitl \
    out_path \
    dataset_path
```

<!-- ## Process and Create Image Pairs

```bash
python datasets/colmap.py \
    --check_images \
    --max_pairs 500 \
    --name eth3d \
    out_path \
    dataset_path 
```

> Example: out_path -> /mnt/data/gg/mdrpbench/pt, dataset_path -> /mnt/data/gg/pt


## Extract Mono Depths

Extract depth with a single MDE model:
```bash
# example - MoGeV2
python depth_estimators/infer_depth.py \
    --model_name MoGeV2 \
    --name eth3d \
    out_path \
    dataset_path
```

## Extract Correspondences

```bash
python matchers/splg.py \
    --name eth3d \
    out_path \
    dataset_path 
``` -->