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


## Extract Mono Depths

Extract depth with a single MDE model:
```bash
# example - MoGeV2
python depth_estimators/infer_depth.py \
    --model_name MoGeV2 \
    --name ETH3D \
    out_path \
    dataset_path
```