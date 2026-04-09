import os
import shutil
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, "./")
from datasets.colmap_utils import read_model, write_model


if __name__ == '__main__':
    root = Path("/mnt/data/gg/benchmarks/ETH3D_depth")
    colmap_root = Path("/mnt/data/gg/eth3d/dslr")
    out_root = Path("/mnt/data/gg/benchmarks/ETH3D")

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    for scene in tqdm(scenes):
        scene_out_root = out_root / scene
        scene_out_root.mkdir(parents=True, exist_ok=True)

        # copy colmap gt
        model_path = colmap_root / scene / "dslr_calibration_undistorted"
        cameras, images, points3d = read_model(model_path, ext=".txt")

        out_model_path = scene_out_root / "gtcolmap"
        out_model_path.mkdir(parents=True, exist_ok=True)
        write_model(cameras, images, points3d, out_model_path, ext='.txt')

        # copy original images
        shutil.copytree(colmap_root / scene / "images" / "dslr_images_undistorted", scene_out_root / "images_original")

        # copy resized images (for depth evaluation)
        images = sorted(list((root / scene).glob("*")))
        images = [image for image in images if image.is_dir()]

        image_folder = scene_out_root / "images"
        depth_folder = scene_out_root / "depths"
        seg_folder = scene_out_root / "segmentations"
        intrinsics_folder = scene_out_root / "intrinsics"
        image_folder.mkdir(parents=True, exist_ok=True)
        depth_folder.mkdir(parents=True, exist_ok=True)
        seg_folder.mkdir(parents=True, exist_ok=True)
        intrinsics_folder.mkdir(parents=True, exist_ok=True)

        for image in images:
            # copy an image
            shutil.copyfile(image / "image.jpg", image_folder / f"{image.stem}.jpg")

            # copy depth map
            shutil.copyfile(image / "depth.png", depth_folder / f"{image.stem}.png")

            # copy segmentation map
            shutil.copyfile(image / "segmentation.png", seg_folder / f"{image.stem}.png")

            # copy intrinsic
            shutil.copyfile(image / "meta.json", intrinsics_folder / f"{image.stem}.json")
        # break