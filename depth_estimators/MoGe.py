import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_estimators.base import BaseDepthEstimator
from moge.model.v2 import MoGeModel


class MoGeV2(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.model = MoGeModel.from_pretrained(checkpoint_name)

        self.num_tokens = None
        self.resolution_level = 9
        self.num_tokens_range = [1200, 3600]

    @property
    def name(self):
        intrinsics = '-known_focal' if self.requires_intrinsics else ''
        return f'MoGeV2-{self.checkpoint_name.split("/")[1]}{intrinsics}'

    @staticmethod
    def upsample(image, h, w):
        return F.interpolate(image, (h, w), mode="bilinear", align_corners=False, antialias=False)

    def infer(self, image, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when MoGe is used with known focal")

        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        # input_image, scale = self.enforce_max_dim(input_image)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.model.device)


        img_h, img_w = input_image.shape[-2:]
        # Determine the number of base tokens to use
        # if self.num_tokens is None:
        #     min_tokens, max_tokens = self.num_tokens_range
        #     num_tokens = int(min_tokens + (self.resolution_level / 9) * (max_tokens - min_tokens))
        # else:
        #     num_tokens = self.num_tokens

        # base_h, base_w = int((num_tokens / aspect_ratio) ** 0.5), int((num_tokens * aspect_ratio) ** 0.5)
        # resized_image = F.interpolate(input_image, (base_h * 14, base_w * 14), mode="bilinear",
        #                               align_corners=False, antialias=not self.model.encoder.onnx_compatible_mode)

        # Infer
        # if self.requires_intrinsics:
        #     # todo calculate
        #     fov_x = ...
        #     output = self.model.infer(resized_image, fov_x = fov_x)
        # else:

        output_original = self.model.infer(input_image, use_fp16=True)
        # output_resized = self.model.infer(resized_image, use_fp16=False)
        # points_resized_back = self.upsample(output_resized['points'].permute(0, 3, 1, 2), img_h, img_w).permute(0, 2, 3, 1)



        inference_K = output_original['intrinsics'].cpu().numpy()[0]
        K = np.copy(inference_K)
        K[0, :] *= img_w
        K[1, :] *= img_h

        depth = output_original['depth'][0].cpu().numpy().astype(np.float16)

        return {'depth': depth, 'inference_K': inference_K, 'K': K}



