from time import perf_counter_ns

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from PIL import Image
from pathlib import Path

from pi3.utils.basic import load_multimodal_data
from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d
from pi3.models.pi3x import Pi3X as Pi3XModel


import sys
sys.path.insert(0, "./")
from depth_estimators.base import BaseDepthEstimator

PIXEL_LIMIT = 255000


def load_and_preprocess_images(image_paths: list[str | Path], require_intrinsics=False, **kwargs):
    sources = []
    for img_path in image_paths:
        sources.append(Image.open(img_path).convert("RGB"))

    # determine a uniform target size for all images based on the first Image
    # this is necessary to ensure all tensors have the same dimensions for stacking
    first_img = sources[0]
    orig_width, orig_height = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (orig_width * orig_height)) if orig_height * orig_width > 0 else 1
    target_width, target_height = orig_width * scale, orig_height * scale
    k, m = round(target_width / 14), round(target_height / 14)

    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > target_width / target_height:
            k -= 1
        else:
            m -= 1

    target_width, target_height = max(1, k) * 14, max(1, m) * 14
    tensor_list, orig_coords = [], []
    TF_transform = TF.ToTensor()

    for img_pil in sources:
        width, height = img_pil.size
        orig_coords.append((height, width))
        resized_img = img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img_tensor = TF_transform(resized_img)
        tensor_list.append(img_tensor)

    K = None
    if require_intrinsics:
        scale_x, scale_y = target_width / width, target_height / height
        K = kwargs['K']
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        K = torch.from_numpy(K).float()[None]

    images_tensor = torch.stack(tensor_list, dim=0)

    return images_tensor, orig_coords, K

class Pi3(BaseDepthEstimator):
    def __init__(self, checkpoint_name="Pi3X", *args, requires_intrinsics=False, **kwargs):
        super().__init__(*args, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.use_multimodal = False

    def load_model(self):
        self.model = Pi3XModel.from_pretrained(f"yyfz233/{self.checkpoint_name}").eval()
        if not self.requires_intrinsics:
            self.model.disable_multimodal()
        self.model = self.model.cuda()

    @property
    def name(self):
        intrinsics = "Calib" if self.requires_intrinsics else ""
        return f'Pi3{intrinsics}-{self.checkpoint_name}'

    def run_model(self, images, orig_resolutions, intrinsics):
        # images (N, 3, H, W)
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        conditions = {}

        with torch.no_grad():
            start_time = perf_counter_ns()
            with torch.amp.autocast("cuda", dtype=self.dtype):
                results = self.model(imgs=images[None], intrinsics=intrinsics)
            runtime = perf_counter_ns() - start_time

        # recover intrinsics from rays_d
        rays_d = F.normalize(results['local_points'], dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
        depth_map = results['local_points'][..., 2]
        depth_conf = results['conf'][..., 0]
        extrinsics = results['camera_poses']

        depth_map, depth_conf = depth_map.squeeze(0), depth_conf.squeeze(0)
        intrinsics, extrinsics = intrinsics.squeeze(0), extrinsics.squeeze(0)
        depth_map, depth_conf, intrinsics = self.upsample_predictions(depth_map, depth_conf, intrinsics, orig_resolutions)
        depth_conf = torch.sigmoid(depth_conf)
        non_edge = ~depth_edge(depth_map, rtol=0.03)
        depth_conf = torch.logical_and(depth_conf, non_edge)
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
        tensor_image, orig_coords, intrinsics = load_and_preprocess_images([image], self.requires_intrinsics, **kwargs)
        tensor_image = tensor_image.cuda()
        if intrinsics is not None:
            intrinsics = intrinsics.cuda()[None]
        depth_map, depth_conf, intrinsic, extrinsic, runtime = self.run_model(tensor_image, orig_coords, intrinsics)
        if depth_map.shape[0] == 1:
            depth_map, depth_conf = depth_map[0], depth_conf[0]
            intrinsic, extrinsic = intrinsic[0], extrinsic[0]
        return {"depth": depth_map, "K": intrinsic, "runtime": runtime}

if __name__ == '__main__':
    image_path = "./assets/kitchen/images/00.png"
    image_path2 = "./assets/kitchen/images/01.png"
    model = Pi3(requires_intrinsics=True)
    model.load_model()
    print(model.name)
    # model.infer([image_path, image_path2])
    K = np.array([
        [500, 0, 389],
        [0, 500, 260],
        [0, 0, 1]
    ]).astype(np.float32)
    model.infer(image_path, K=K)
