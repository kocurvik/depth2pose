import math
from pathlib import Path

import cv2
import numpy as np
import pipeline
import torch
import utils3d
from PIL import Image

from .io import read_depth, read_image, read_json


def focal_to_fov_numpy(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def norm3d(x: np.ndarray) -> np.ndarray:
    "Faster `np.linalg.norm(x, axis=-1)` for 3D vectors"
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2]))


class EvalDataLoaderPipeline:
    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        drop_max_depth: float = 1000.0,
        num_load_workers: int = 4,
        num_process_workers: int = 8,
        depth_unit: str = None,
    ):
        self.width, self.height = width, height
        self.drop_max_depth = drop_max_depth
        self.path = Path(path)
        self.filenames = self.read_filenames()
        self.depth_unit = depth_unit

        self.rng = np.random.default_rng(seed=0)

        self.pipeline = pipeline.Sequential(
            [
                self._generator,
                pipeline.Parallel([self._load_instance] * num_load_workers),
                pipeline.Parallel([self._process_instance] * num_process_workers),
                pipeline.Buffer(4),
            ]
        )

    def read_filenames(self):
        scenes = sorted(list(self.path.glob("*")))
        scenes = [scene.stem for scene in scenes if scene.is_dir()]

        filenames = []
        for scene in scenes:
            files = sorted(list((self.path / scene / "images").glob("*.jpg")))
            files = [f"{scene}/{file.stem}" for file in files]
            filenames.extend(files)
        return filenames

    def __len__(self):
        return math.ceil(len(self.filenames))

    def _generator(self):
        for idx in range(len(self)):
            yield idx

    def _load_instance(self, idx):
        if idx >= len(self.filenames):
            return None

        scene, filename = self.filenames[idx].split("/")

        image = read_image(self.path / scene / "images" / f"{filename}.jpg")
        depth = read_depth(self.path / scene / "depths" / f"{filename}.png")
        meta = read_json(self.path / scene / "intrinsics" / f"{filename}.json")

        instance = {
            "scenename": scene,
            "filename": filename,
            "width": self.width,
            "height": self.height,
            "image": image,
            "depth": np.nan_to_num(depth, nan=1, posinf=1, neginf=1),
            "depth_mask": np.isfinite(depth),
            "intrinsics": np.array(meta["intrinsics"], dtype=np.float32),
        }
        return instance

    def _process_instance(self, instance: dict):
        if instance is None:
            return None

        image, depth, depth_mask, intrinsics = (
            instance["image"],
            instance["depth"],
            instance["depth_mask"],
            instance["intrinsics"],
        )

        raw_height, raw_width = image.shape[:2]
        raw_horizontal, raw_vertical = (
            abs(1.0 / intrinsics[0, 0]),
            abs(1.0 / intrinsics[1, 1]),
        )
        raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
        tgt_width, tgt_height = instance["width"], instance["height"]
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
        corners = np.concatenate(
            [corners, np.ones((4, 1), dtype=np.float32)], axis=1
        ) @ (np.linalg.inv(intrinsics).T @ R.T)  # corners in viewport's camera plane
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
        tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(
            fx, fy, 0.5, 0.5
        ).astype(np.float32)

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
        distance = norm3d(
            utils3d.np.depth_map_to_point_map(depth, intrinsics=intrinsics)
        )

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
        tgt_depth_mask = (
            cv2.remap(
                depth_mask.astype(np.uint8),
                pixel_remap[:, :, 0],
                pixel_remap[:, :, 1],
                cv2.INTER_NEAREST,
            )
            > 0
        )

        # drop depth greater than drop_max_depth
        max_depth = (
            np.nanquantile(np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01)
            * self.drop_max_depth
        )
        tgt_depth_mask &= tgt_depth <= max_depth
        tgt_depth = np.nan_to_num(tgt_depth, nan=0.0)

        if self.depth_unit is not None:
            tgt_depth *= self.depth_unit

        if not np.any(tgt_depth_mask):
            # always make sure that mask is not empty, otherwise the loss calculation will crash
            tgt_depth_mask = np.ones_like(tgt_depth_mask)
            tgt_depth = np.ones_like(tgt_depth)
            instance["label_type"] = "invalid"

        instance.update(
            {
                # "image": torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(
                #     2, 0, 1
                # ),
                "depth": torch.from_numpy(tgt_depth).float(),
                "depth_mask": torch.from_numpy(tgt_depth_mask).bool(),
                "intrinsics": torch.from_numpy(tgt_intrinsics).float(),
                "is_metric": self.depth_unit is not None,
            }
        )

        instance = {k: v for k, v in instance.items() if v is not None}

        return instance

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get(self):
        return self.pipeline.get()
