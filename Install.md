```bash
# these will install latest torch version
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install git+https://github.com/microsoft/MoGe.git
pip install --no-deps git+https://github.com/lpiccinelli-eth/UniDepth
# this will reinstall torch to 2.4.1 so UniDepth works
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu124

# for Metric3D
mamba install mmengine mmcv --no-deps
pip install --no-deps git+https://github.com/kocurvik/UniK3D
pip install --no-deps git+https://github.com/apple/ml-depth-pro
mkdir -p checkpoints
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints
# using this PR: https://github.com/DepthAnything/Depth-Anything-V2/pull/159
pip install --no-deps git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" -O checkpoints/depth_anything_v2_vits.pth
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" -O checkpoints/depth_anything_v2_vitb.pth
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" -O checkpoints/depth_anything_v2_vitl.pth

# poselib with all solvers
pip install git+https://github.com/kocurvik/PoseLib.git@mdrpbench
```
