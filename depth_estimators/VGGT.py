from time import perf_counter_ns

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from PIL import Image
from pathlib import Path

# import sys
# sys.path.insert(0, "./")
from depth_estimators.base import BaseDepthEstimator
from vggt.models.vggt import VGGT as VGGTBase
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_and_preprocess_images(image_path_list, target_size=518, mode="pad"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images, orig_coords = [], []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        content_h = img.shape[1]
        content_w = img.shape[2]
        pad_top = pad_left = 0

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        orig_coords.append((height, width, content_h, content_w, pad_top, pad_left))

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images, orig_coords

class VGGT(BaseDepthEstimator):
    def __init__(self, checkpoint_name="VGGT-1B", *args, max_dim=None, requires_intrinsics=False, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.vggt_fixed_resolution = 518
        self.img_load_resolution = 1024

    def load_model(self):
        self.model = VGGTBase()
        _URL = f"https://huggingface.co/facebook/{self.checkpoint_name}/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.cuda()

    @property
    def name(self):
        return f'VGGT-{self.checkpoint_name}'

    def run_VGGT(self, images, orig_resolutions):
        # images (B, 3, H, W)
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        with torch.no_grad():
            start_time = perf_counter_ns()
            with torch.amp.autocast("cuda", dtype=self.dtype):
                images = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            runtime = perf_counter_ns() - start_time

        depth_map, depth_conf = depth_map.squeeze(0).squeeze(-1), depth_conf.squeeze(0)
        intrinsic, extrinsic = intrinsic.squeeze(0), extrinsic.squeeze(0)
        depth_map, depth_conf, intrinsic = self.upsample_predictions(depth_map, depth_conf, intrinsic, orig_resolutions)
        return depth_map.cpu().numpy(), depth_conf.cpu().numpy(), intrinsic.cpu().numpy(), extrinsic.cpu().numpy(), runtime

    def upsample_predictions(self, depth_maps, depth_confs, intrinsics, orig_coords):
        new_depth_maps, new_depth_confs = [], []
        for i in range(len(depth_maps)):
            height, width, content_h, content_w, pad_top, pad_left = orig_coords[i]
            depth_map = depth_maps[i][pad_top:pad_top + content_h, pad_left:pad_left + content_w]
            depth_conf = depth_confs[i][pad_top:pad_top + content_h, pad_left:pad_left + content_w]
            intrinsics[i][0, 2] = width / 2
            intrinsics[i][1, 2] = height / 2
            intrinsics[i][0, 0] *= width / content_w
            intrinsics[i][1, 1] *= height / content_h
            depth_map = F.interpolate(depth_map[None, None], size=(height, width), mode='bilinear', align_corners=False).squeeze()
            depth_conf = F.interpolate(depth_conf[None, None], size=(height, width), mode='bilinear', align_corners=False).squeeze()
            new_depth_maps.append(depth_map)
            new_depth_confs.append(depth_conf)
        return torch.stack(new_depth_maps), torch.stack(new_depth_confs), intrinsics

    def infer(self, image, size=None, **kwargs):
        tensor_image, orig_coords = load_and_preprocess_images([image], self.vggt_fixed_resolution)
        tensor_image = tensor_image.cuda()
        depth_map, depth_conf, intrinsic, extrinsic, runtime = self.run_VGGT(tensor_image, orig_coords)
        if depth_map.shape[0] == 1:
            depth_map, depth_conf = depth_map[0], depth_conf[0]
            intrinsic, extrinsic = intrinsic[0], extrinsic[0]
        return {"depth": depth_map, "K": intrinsic, "runtime": runtime}


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


if __name__ == '__main__':
    image_path = "00001.png"
    model = VGGT()
    model.load_model()
    print(model.name)
    # model.infer([image_path, image_path2])
    out = model.infer(image_path)
    depth_colored =  colorize_depth(out['depth'])
    cv2.imwrite("colorized.png", depth_colored)

