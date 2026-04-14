import shutil
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "./")
from datasets.colmap_utils import Camera, Point3D, write_model, rotmat2qvec
from datasets.colmap_utils import Image as CImage

SPRING_BASELINE = 0.065


def disparity_to_metric_depth(disparity, fx, baseline=SPRING_BASELINE):
    return fx * baseline / disparity


def read_depth_spring(depth_file, intrinsic):
    with h5py.File(depth_file, "r") as f:
        disparity = f["disparity"][()]
    disparity = disparity[::2, ::2]
    # disparity = np.nan_to_num(disparity, 0.0)
    return disparity_to_metric_depth(disparity, intrinsic[0])


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
    root = Path("/mnt/data/gg/benchmarks_original/spring")
    out_root = Path("/mnt/data/gg/benchmarks/spring")

    if out_root.exists():
        shutil.rmtree(out_root)

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    h, w = 1080, 1920
    sample_rate = 5
    for scene in scenes:
        print(scene)
        image_folder_path = root / scene / "frame_left"
        depth_folder_path = root / scene / "disp1_left"
        cam_folder_path = root / scene / "cam_data"

        save_model_path = out_root / scene / "colmap_gt"
        save_image_folder = out_root / scene / "images"
        save_depth_folder = out_root / scene / "depths_gt"
        save_model_path.mkdir(parents=True, exist_ok=True)
        save_image_folder.mkdir(parents=True, exist_ok=True)
        save_depth_folder.mkdir(parents=True, exist_ok=True)

        # read camera data
        intrinsics = np.loadtxt(cam_folder_path / "intrinsics.txt").astype(np.float32)
        extrinsics = np.loadtxt(cam_folder_path / "extrinsics.txt").astype(np.float32)

        # save new images and depths
        # filenames = sorted([Path(image.name) for image in images.values()])
        filenames = sorted(list(image_folder_path.glob("*.png")))
        filenames = [filename.stem.split("_")[-1] for filename in filenames]
        # subsample frames
        filenames = filenames[::sample_rate]
        intrinsics = intrinsics[::sample_rate]
        extrinsics = extrinsics[::sample_rate]

        cameras, images, points3ds = {}, {}, {}

        for i in tqdm(range(len(filenames))):
            filename = filenames[i]
            # create colmap model
            intrinsic, extrinsic = intrinsics[i], extrinsics[i].reshape(4, 4)
            print(extrinsic.shape)
            # copy image file
            shutil.copyfile(
                image_folder_path / f"frame_left_{filename}.png",
                save_image_folder / f"{filename}.png",
            )

            depth = read_depth_spring(
                depth_folder_path / f"disp1_left_{filename}.dsp5", intrinsic
            )
            np.savez_compressed(
                save_depth_folder / filename, depth=depth.astype(np.float32)
            )
            cameras[i] = Camera(
                id=i,
                model="PINHOLE",
                width=w,
                height=h,
                params=intrinsic,
            )
            images[i] = CImage(
                id=i,
                qvec=rotmat2qvec(extrinsic[:3, :3]),
                tvec=extrinsic[:3, -1],
                camera_id=i,
                name=f"{filename}.png",
                xys=np.array([]),
                point3D_ids=np.array([])
            )
            points3ds[i] = Point3D(
                id=0,
                xyz=np.array([]),
                rgb=np.array([]),
                error=0.0,
                image_ids=np.array([]),
                point2D_idxs=np.array([])
            )
            # break

        # save a new colmap model
        write_model(cameras, images, points3ds, save_model_path, ext=".txt")

        break
