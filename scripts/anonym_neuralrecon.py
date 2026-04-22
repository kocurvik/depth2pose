import subprocess
import numpy as np
from pathlib import Path
from PIL import Image

"""
Installation instructions to use easyanon

pip install hatchling
pip install git+https://github.com/spatial-intelligence-group/easy_anon.git --no-build-isolation
"""

img_extensions = ['.png', '.jpg', '.jpeg', 'heif', 'heic']

def mask_out_image(image_path, mask_save_dir, image_save_dir):
    subprocess.call([
        "easy-mask",
        image_path,
        mask_save_dir,
        "--labels",
        "person",

    ])
    subprocess.call([
        "easy-anon",
        image_path,
        mask_save_dir,
        image_save_dir,
        "--infill_mode",
        "single_color",

    ])

def filter_depth_mask(image_path: Path, mask_save_dir: Path, depth_gt_dir: Path, depth_mask_dir: Path):
    image_fname = image_path.stem
    mask = Image.open(mask_save_dir / f"{image_fname}.png")
    mask = np.asarray(mask).astype(bool)
    bg_mask = ~mask

    depth_gt = np.load(depth_mask_dir / f"{image_fname}.npz")['depth']
    depth_gt[bg_mask] = 0.0
    np.savez_compressed(
        depth_gt_dir / image_fname, depth=depth_gt.astype(np.float32)
    )


def run_on_scenes(root: Path, scenes: list[str], selected_scene: str | None = None):
    for scene in scenes:
        print(f"Running on {scene}")
        scene_dir = root / scene
        image_dir = scene_dir / "images"
        mask_save_dir = scene_dir / "masks"
        image_save_dir = scene_dir / "images_masked"
        depth_gt_dir = scene_dir / "depths_gt"
        depth_mask_dir = scene_dir / "depths_masked"
        mask_save_dir.mkdir(parents=True, exist_ok=True)
        image_save_dir.mkdir(parents=True, exist_ok=True)

        subprocess.call(["mv", depth_gt_dir, depth_mask_dir])

        depth_gt_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            continue

        image_paths = sorted([x for x in image_dir.iterdir() if x.suffix.lower() in img_extensions])

        for image_path in image_paths:
            mask_out_image(image_path, mask_save_dir, image_save_dir)
            filter_depth_mask(image_path, mask_save_dir, depth_gt_dir, depth_mask_dir)
            # break

        # subprocess.call(["rm -rf", depth_mask_dir])
        # break


if __name__ == '__main__':
    root = Path("/mnt/data/gg/benchmarks/heritage_recon")
    scenes = sorted(list(root.glob("*")))
    # scenes = [scene.stem for scene in scenes if scene.is_dir()]
    scenes = ["palacio_de_bellas_artes"]
    run_on_scenes(root, scenes)
