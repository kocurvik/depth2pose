# Installation Guide

This document provides step-by-step instructions for setting up the environment required to run this project. The setup involves multiple depth estimation models and related dependencies.

## Prerequisites

- Python 3.8 or higher
- CUDA 12.4 compatible GPU (recommended)
- `mamba` or `conda` (for conda-based installations)
- Git

## Step 0

Recreate the saved mamba/conda environment, then install the pip dependencies into it:

```bash
mamba env create -f environment.yml
mamba activate mdrpbench
pip install -r requirements.txt
```

If the environment already exists, update it instead:

```bash
mamba env update -f environment.yml --prune
```

### Step 1: Initial Depth Model Dependencies

Install the following depth estimation models. Use `--no-deps` for packages with their own torch requirements to preserve PyTorch 2.4.1:

```bash
pip install --no-deps git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install --no-deps git+https://github.com/microsoft/MoGe.git
```
### Step 2
Install PyTorch 2.4.1 with CUDA 12.4 support first. This version is required for proper compatibility with the depth models:
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu124
```
Torch will be installed also in the previous step, but we need a version compatible with all MDE implementations and matchers.


### Step 3: LoMa Installation

```bash
pip install git+https://github.com/davnords/LoMa.git
```

### Step 4: Metric3D and Related Models

Install MMEngine and MMCv dependencies:

```bash
mamba install mmengine mmcv --no-deps
```

Install additional depth models:

```bash
pip install git+https://github.com/kocurvik/UniK3D
pip install git+https://github.com/apple/ml-depth-pro
pip install git+https://github.com/lpiccinelli-eth/UniDepth
```

### Step 5: Pre-trained Model Checkpoints

Create a checkpoints directory and download required model weights:

```bash
mkdir -p checkpoints
```

**Depth Pro (Apple):**

```bash
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints
```

**Depth Anything V2:**

```bash
pip install git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" -O checkpoints/depth_anything_v2_vits.pth
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" -O checkpoints/depth_anything_v2_vitb.pth
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" -O checkpoints/depth_anything_v2_vitl.pth
```

### Step 6: InfiniDepth Installation

```bash
pip install git+https://github.com/kocurvik/InfiniDepth
wget https://huggingface.co/ritianyu/InfiniDepth/resolve/main/infinidepth.ckpt -O checkpoints/infinidepth.ckpt
```

### Step 7: Additional Models

Install VGGT (Facebook Research):

```bash
pip install git+https://github.com/facebookresearch/vggt.git
```

Install Pi3:

```bash
pip install git+https://github.com/yyfz/Pi3.git
```

Install MapAnything:

```bash
pip install git+https://github.com/facebookresearch/map-anything.git
```

### Step 8: PoseLib Installation

Install the specialized PoseLib version with all solvers:

```bash
pip install git+https://github.com/kocurvik/PoseLib.git@mdrpbench
```

## Complete Installation Script

If you prefer to run the entire installation at once, save the following as a shell script and execute it:

```bash
#!/bin/bash
set -e  # Exit on error

mamba env create -f environment.yml
mamba activate mdrpbench
pip install -r requirements.txt

echo "Installing initial depth models..."
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install git+https://github.com/microsoft/MoGe.git
pip install git+https://github.com/lpiccinelli-eth/UniDepth

echo "Installing PyTorch 2.4.1 with CUDA 12.4 support..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu124

echo "Installing LoMa..."
pip install git+https://github.com/davnords/LoMa.git

echo "Installing Metric3D dependencies..."
mamba install mmengine mmcv --no-deps
pip install git+https://github.com/kocurvik/UniK3D
pip install git+https://github.com/apple/ml-depth-pro

echo "Downloading checkpoints..."
mkdir -p checkpoints
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" -O checkpoints/depth_anything_v2_vits.pth
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" -O checkpoints/depth_anything_v2_vitb.pth
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" -O checkpoints/depth_anything_v2_vitl.pth

echo "Installing Depth Anything V2..."
pip install git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1

echo "Installing InfiniDepth..."
pip install git+https://github.com/kocurvik/InfiniDepth
wget https://huggingface.co/ritianyu/InfiniDepth/resolve/main/infinidepth.ckpt -O checkpoints/infinidepth.ckpt

echo "Installing additional models..."
pip install git+https://github.com/facebookresearch/vggt.git
pip install git+https://github.com/yyfz/Pi3.git
pip install git+https://github.com/facebookresearch/map-anything.git

echo "Installing PoseLib..."
pip install git+https://github.com/kocurvik/PoseLib.git@mdrpbench

echo "Installation complete!"
```

## Troubleshooting

### PyTorch Version Conflicts

If you encounter issues with PyTorch versions, verify your installation:

```bash
python -c "import torch; print(torch.__version__)"
```

You should see `2.4.1`. If not, reinstall using the command in Step 2.


### Missing Checkpoints

If checkpoint downloads fail, you can manually download them from the provided URLs and place them in the `checkpoints/` directory.

### Dependency Issues

If you encounter dependency conflicts, try installing packages individually rather than using the complete script. This allows you to identify which package is causing the issue.

## Verifying Installation

After installation, verify that all models are accessible:

```bash
python -c "import torch; import depth_anything; import moge" 
echo "Core dependencies installed successfully!"
```

## Notes

- PyTorch 2.4.1 is installed first and pinned explicitly
- `--no-deps` is used selectively for packages with conflicting torch dependencies (Depth-Anything-3 and MoGe) to prevent torch overrides
- Other packages install normally with their dependencies, which respect the pinned PyTorch version
- The mamba command for MMEngine and MMCv uses `--no-deps` since they're from conda and don't need pip dependencies
- Checkpoint files (`.pt` and `.ckpt`) require sufficient disk space (~5-10 GB total)
- Some models may require additional setup; refer to their respective GitHub repositories for more details