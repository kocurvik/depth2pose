# Benchmarking MDE models

## Installation

```bash
conda env create --file environment.yml
conda activate mdrpbench
conda install --file requirements.txt


conda crate -n mdrpbench python=3.10
conda activate mdrpbench
conda install -c "nvidia/label/cuda-2.8" cuda-toolkit
pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@41736238f5bced4debf3f2a12375d2466874866d

# install moge
pip install --no-build-isolation git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8
```