import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import utils3d
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "./")
from datasets.colmap_utils import Camera, read_model, write_model
from datasets.colmap_utils import Image as CImage
from datasets.io import read_depth, read_image, read_json


def norm3d(x: np.ndarray) -> np.ndarray:
    "Faster `np.linalg.norm(x, axis=-1)` for 3D vectors"
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2]))


def process_depth_image(image, depth, intrinsics, h, w):
    depth_mask = np.isfinite(depth)
    raw_height, raw_width = image.shape[:2]
    raw_horizontal, raw_vertical = (
        abs(1.0 / intrinsics[0, 0]),
        abs(1.0 / intrinsics[1, 1]),
    )
    raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
    tgt_width, tgt_height = w, h
    tgt_aspect = tgt_width / tgt_height

    # set expected target view field
    tgt_horizontal = min(raw_horizontal, raw_vertical * tgt_aspect)
    tgt_vertical = tgt_horizontal / tgt_aspect

    # set target view direction
    cu, cv = 0.5, 0.5
    direction = utils3d.np.unproject_cv(
        np.array([[cu, cv]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        intrinsics=intrinsics,
    )[0]
    R = utils3d.np.rotation_matrix_from_vectors(
        direction, np.array([0, 0, 1], dtype=np.float32)
    )

    # restrict target view field within the raw view
    corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (
        np.linalg.inv(intrinsics).T @ R.T
    )  # corners in viewport's camera plane
    corners = corners[:, :2] / corners[:, 2:3]

    warp_horizontal, warp_vertical = (
        abs(1.0 / intrinsics[0, 0]),
        abs(1.0 / intrinsics[1, 1]),
    )
    for i in range(4):
        intersection, _ = utils3d.np.ray_intersection(
            np.array([0.0, 0.0]),
            np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
            corners[i - 1],
            corners[i] - corners[i - 1],
        )
        warp_horizontal, warp_vertical = (
            min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()),
            min(warp_vertical, 2 * np.abs(intersection[:, 1]).min()),
        )
    tgt_horizontal, tgt_vertical = (
        min(tgt_horizontal, warp_horizontal),
        min(tgt_vertical, warp_vertical),
    )

    # get target view intrinsics
    fx, fy = 1.0 / tgt_horizontal, 1.0 / tgt_vertical
    tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(
        np.float32
    )

    # do homogeneous transformation with the rotation and intrinsics
    # 4.1 The image and depth is resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
    tgt_pixel_w, tgt_pixel_h = (
        tgt_horizontal / tgt_width,
        tgt_vertical / tgt_height,
    )  # (should be exactly the same for x and y axes)
    rescaled_w, rescaled_h = (
        int(raw_width * raw_pixel_w / tgt_pixel_w),
        int(raw_height * raw_pixel_h / tgt_pixel_h),
    )
    image = np.array(
        Image.fromarray(image).resize(
            (rescaled_w, rescaled_h), Image.Resampling.LANCZOS
        )
    )

    depth, depth_mask = utils3d.np.masked_nearest_resize(
        depth, mask=depth_mask, size=(rescaled_h, rescaled_w)
    )
    distance = norm3d(utils3d.np.depth_map_to_point_map(depth, intrinsics=intrinsics))

    # 4.2 calculate homography warping
    transform = intrinsics @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
    uv_tgt = utils3d.np.uv_map(tgt_height, tgt_width)
    pts = (
        np.concatenate(
            [uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1
        )
        @ transform.T
    )
    uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
    pixel_remap = utils3d.np.uv_to_pixel(uv_remap, (rescaled_h, rescaled_w)).astype(
        np.float32
    )

    tgt_image = cv2.remap(
        image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR
    )
    tgt_distance = cv2.remap(
        distance, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST
    )
    tgt_ray_length = utils3d.np.unproject_cv(
        uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics
    )
    tgt_ray_length = (
        tgt_ray_length[:, :, 0] ** 2
        + tgt_ray_length[:, :, 1] ** 2
        + tgt_ray_length[:, :, 2] ** 2
    ) ** 0.5
    tgt_depth = tgt_distance / (tgt_ray_length + 1e-12)
    return tgt_image, tgt_depth


def update_colmap_cameras(cameras, h, w):
    new_cameras = {}
    for cam_id, camera in cameras.items():
        sh, sw = h / camera.height, w / camera.width
        params = camera.params
        params[0] *= sw
        params[1] *= sh
        params[2] *= sw
        params[3] *= sh
        new_camera = Camera(
            id=camera.id, model=camera.model, width=w, height=h, params=params
        )
        new_cameras[cam_id] = new_camera
    return new_cameras


def update_colmap_images(images, cameras, h, w):
    new_images = {}
    for image_id, image in images.items():
        cam = cameras[image.camera_id]
        sh, sw = h / cam.height, w / cam.width
        new_xys = image.xys
        new_xys[:, 0] *= sw
        new_xys[:, 1] *= sh
        new_image = CImage(
            id=image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=f"{Path(image.name).stem}.jpg",
            xys=new_xys,
            point3D_ids=image.point3D_ids,
        )
        new_images[image_id] = new_image
    return new_images


if __name__ == "__main__":
    root = Path("/mnt/data/gg/benchmarks_original/ETH3D_depth")
    colmap_root = Path("/mnt/data/gg/benchmarks_original/eth3d/dslr")
    out_root = Path("/mnt/data/gg/benchmarks/ETH3D")

    if out_root.exists():
        shutil.rmtree(out_root)

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    h, w = 1365, 2048

    for scene in tqdm(scenes):
        scene_out_root = out_root / scene
        scene_out_root.mkdir(parents=True, exist_ok=True)

        # copy colmap gt
        model_path = colmap_root / scene / "dslr_calibration_undistorted"
        cameras, images, points3d = read_model(model_path, ext=".txt")
        images = update_colmap_images(images, cameras, h, w)
        cameras = update_colmap_cameras(cameras, h, w)
        out_model_path = scene_out_root / "colmap_gt"
        out_model_path.mkdir(parents=True, exist_ok=True)
        write_model(cameras, images, points3d, out_model_path, ext=".txt")

        # # copy original images
        # shutil.copytree(
        #     colmap_root / scene / "images" / "dslr_images_undistorted",
        #     scene_out_root / "images_original",
        # )

        # copy resized images (for depth evaluation)
        images = sorted(list((root / scene).glob("*")))
        images = [image for image in images if image.is_dir()]

        image_folder = scene_out_root / "images"
        depth_folder = scene_out_root / "depths_gt"
        seg_folder = scene_out_root / "segmentations"
        intrinsics_folder = scene_out_root / "intrinsics"
        image_folder.mkdir(parents=True, exist_ok=True)
        depth_folder.mkdir(parents=True, exist_ok=True)
        seg_folder.mkdir(parents=True, exist_ok=True)
        intrinsics_folder.mkdir(parents=True, exist_ok=True)

        for image in images:
            img = read_image(image / "image.jpg")
            depth = read_depth(image / "depth.png")
            meta = read_json(image / "meta.json")
            intrinsic = np.array(meta["intrinsics"], dtype=np.float32)

            img, depth = process_depth_image(img, depth, intrinsic, h, w)

            # save as a new image
            cv2.imwrite(str(image_folder / f"{image.stem}.jpg"), img[..., ::-1])

            # save depth as np array
            # np.save(depth_folder / f"{image.stem}.npy", depth.astype(np.float32))
            np.savez_compressed(
                depth_folder / f"{image.stem}", depth=depth.astype(np.float32)
            )

            # shutil.copyfile(image / "image.jpg", image_folder / f"{image.stem}.jpg")
            # shutil.copyfile(image / "depth.png", depth_folder / f"{image.stem}.png")

            # copy segmentation map
            shutil.copyfile(
                image / "segmentation.png", seg_folder / f"{image.stem}.png"
            )

            # copy intrinsic
            shutil.copyfile(
                image / "meta.json", intrinsics_folder / f"{image.stem}.json"
            )
        # break
