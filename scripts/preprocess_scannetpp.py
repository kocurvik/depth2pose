import shutil
import sys
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "./")
from datasets.colmap_utils import Camera, read_model, write_model
from datasets.colmap_utils import Image as CImage


def update_colmap_cameras(cameras):
    new_cameras = {}
    for cam_id, camera in cameras.items():
        if camera.id != 2:
            continue
        new_camera = Camera(
            id=0,
            model=camera.model,
            width=camera.width,
            height=camera.height,
            params=camera.params,
        )
        new_cameras[cam_id] = new_camera
    return new_cameras


def update_colmap_images(images):
    new_images = {}
    for image_id, image in images.items():
        if "iphone" not in image.name:
            continue
        new_image = CImage(
            id=image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=0,
            name=f"{Path(image.name).stem}.jpg",
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )
        new_images[image_id] = new_image
    return new_images


if __name__ == "__main__":
    root = Path("/mnt/data/gg/benchmarks_original/scannetpp")
    out_root = Path("/mnt/data/gg/benchmarks/scannetpp")

    if out_root.exists():
        shutil.rmtree(out_root)

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    h, w = 1440, 1920
    for scene in scenes:
        print(scene)
        colmap_model_path = (
            root / scene / "merge_dslr_iphone" / "colmap" / "sparse_render_rgb"
        )
        image_folder_path = root / scene / "merge_dslr_iphone" / "images" / "iphone"
        depth_folder_path = root / scene / "merge_dslr_iphone" / "render_depth"

        # load and process colmap model
        cameras, images, points3d = read_model(colmap_model_path, ext=".bin")
        cameras = update_colmap_cameras(cameras)
        images = update_colmap_images(images)

        save_model_path = out_root / scene / "colmap_gt"
        save_image_folder = out_root / scene / "images"
        save_depth_folder = out_root / scene / "depths_gt"
        save_model_path.mkdir(parents=True, exist_ok=True)
        save_image_folder.mkdir(parents=True, exist_ok=True)
        save_depth_folder.mkdir(parents=True, exist_ok=True)

        # save a new colmap model
        write_model(cameras, images, points3d, save_model_path, ext=".txt")

        # save new images and depths
        filenames = sorted([Path(image.name) for image in images.values()])
        for filename in tqdm(filenames):
            depth = (
                imageio.imread(depth_folder_path / f"{filename.stem}.png") / 1000.0
            )  # mm to meters

            shutil.copyfile(image_folder_path / filename, save_image_folder / filename)

            np.savez_compressed(
                save_depth_folder / f"{filename.stem}", depth=depth.astype(np.float32)
            )

        # break
