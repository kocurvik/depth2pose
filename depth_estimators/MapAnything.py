from time import perf_counter_ns

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from PIL import Image
from pathlib import Path

from mapanything.models import MapAnything as MapAnythingModel
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.image import find_closest_aspect_ratio, IMAGE_NORMALIZATION_DICT


import sys
sys.path.insert(0, "./")
from depth_estimators.base import BaseDepthEstimator

PIXEL_LIMIT = 255000


def load_and_preprocess_images(image_paths: list[str | Path], require_intrinsics=False, **kwargs):
    sources, aspect_ratios = [], []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        aspect_ratios.append(width / height)
        sources.append(img)

    resolution_set = 518
    # calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    target_width, target_height = find_closest_aspect_ratio(average_aspect_ratio, resolution_set)
    target_size = (target_width, target_height)

    img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    ImgNorm = TF.Compose([
        TF.ToTensor(),
        TF.Normalize(mean=img_norm.mean, std=img_norm.std)
    ])
    K = None
    if require_intrinsics:
        scale_x, scale_y = target_width / width, target_height / height
        K = kwargs['K']
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        K = torch.from_numpy(K).float()[None]
    views, orig_coords = [], []
    for img_pil in sources:
        width, height = img_pil.size
        orig_coords.append((height, width))
        resized_img = crop_resize_if_necessary(img_pil, resolution=target_size)[0]
        img_tensor = ImgNorm(resized_img)[None]
        views.append({
            "img": img_tensor,
            "data_norm_type": ["dinov2"],
        })
        if require_intrinsics:
            views[-1]["intrinsics"] = K
    return views, orig_coords

class MapAnything(BaseDepthEstimator):
    def __init__(self, checkpoint_name="map-anything", *args, requires_intrinsics=False, **kwargs):
        super().__init__(*args, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.use_multimodal = False

    def load_model(self):
        self.model = MapAnythingModel.from_pretrained(f"facebook/{self.checkpoint_name}")
        self.model = self.model.cuda()

    @property
    def name(self):
        intrinsics = "Calib" if self.requires_intrinsics else ""
        return f'MapAnything{intrinsics}-{self.checkpoint_name}'

    def run_model(self, images, orig_resolutions):
        start_time = perf_counter_ns()
        results = self.model.infer(
            images,
            memory_efficient_inference=True,
            minibatch_size=None,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,    # apply masking to dense geometry outputs
            mask_edges=True,    # remove edge artifacts by using normals and depth
            apply_confidence_mask=False,    # filter low confidence regions
            confidence_percentile=10,       # remove bottom 10 percentile confidence pixels
            use_multiview_confidence=False, # enable multi-view depth consistency based confidence in place of learning based one
            ignore_calibration_inputs=not self.requires_intrinsics,
        )[0]
        runtime = perf_counter_ns() - start_time
        depth_maps, depth_confs = results['depth_z'].squeeze(-1), results['conf']
        intrinsics = results['intrinsics']
        camera_poses = results['camera_poses']  # in cam2world format
        extrinsics = torch.inverse(camera_poses)

        depth_map, depth_conf, intrinsics = self.upsample_predictions(depth_maps, depth_confs, intrinsics.clone(), orig_resolutions)
        return depth_map.cpu().numpy(), depth_conf.cpu().numpy(), intrinsics.cpu().numpy(), extrinsics.cpu().numpy(), runtime

    def upsample_predictions(self, depth_maps, depth_confs, intrinsics, orig_coords):
        new_depth_maps, new_depth_confs = [], []
        for i in range(len(depth_maps)):
            new_height, new_width = depth_confs[i].shape
            height, width = orig_coords[i]
            intrinsics[i][0, 2] = width/2
            intrinsics[i][1, 2] = height/2
            intrinsics[i][0, 0] *= width / new_width
            intrinsics[i][1, 1] *= height / new_height
            depth_map = F.interpolate(depth_maps[i][None, None], size=orig_coords[i], mode='bilinear', align_corners=False).squeeze()
            depth_conf = F.interpolate(depth_confs[i][None, None], size=orig_coords[i], mode='bilinear', align_corners=False).squeeze()
            new_depth_maps.append(depth_map)
            new_depth_confs.append(depth_conf)
        return torch.stack(new_depth_maps), torch.stack(new_depth_confs), intrinsics

    def infer(self, image: str | Path, size=None, **kwargs):
        tensor_image, orig_coords = load_and_preprocess_images([image], self.requires_intrinsics, **kwargs)
        depth_map, depth_conf, intrinsic, extrinsic, runtime = self.run_model(tensor_image, orig_coords)
        if depth_map.shape[0] == 1:
            depth_map, depth_conf = depth_map[0], depth_conf[0]
            intrinsic, extrinsic = intrinsic[0], extrinsic[0]
        return {"depth": depth_map, "K": intrinsic, "runtime": runtime}

if __name__ == '__main__':
    image_path = "./assets/kitchen/images/00.png"
    image_path2 = "./assets/kitchen/images/01.png"
    model = MapAnything(requires_intrinsics=True)
    model.load_model()
    print(model.name)
    # model.infer([image_path, image_path2])
    K = np.array([
        [500, 0, 389],
        [0, 500, 260],
        [0, 0, 1]
    ]).astype(np.float32)
    model.infer(image_path, K=K)
