import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_estimators.base import BaseDepthEstimator
from moge.model.v1 import MoGeModel as MoGeModelV1
from moge.model.v2 import MoGeModel as MoGeModelV2


class MoGe(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name

        self.version = version

        if version == 2:
            self.model = MoGeModelV2.from_pretrained(checkpoint_name)
        elif version == 1:
            self.model = MoGeModelV1.from_pretrained(checkpoint_name)
        else:
            raise ValueError('MoGe version must be 1 or 2')

        self.num_tokens = None
        self.resolution_level = 9
        self.num_tokens_range = [1200, 3600]

    @property
    def name(self):
        intrinsics = '-known_focal' if self.requires_intrinsics else ''
        return f'MoGeV{self.version}-{self.checkpoint_name.split("/")[1]}{intrinsics}'

    @staticmethod
    def upsample(image, h, w):
        return F.interpolate(image, (h, w), mode="bilinear", align_corners=False, antialias=False)

    def infer(self, image, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when MoGe is used with known focal")

        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        img_h, img_w = input_image.shape[-2:]
        output_original = self.model.infer(input_image, use_fp16=True)
        inference_K = output_original['intrinsics'].cpu().numpy()[0]
        K = np.copy(inference_K)
        K[0, :] *= img_w
        K[1, :] *= img_h
        depth = output_original['depth'][0].cpu().numpy()

        return {'depth': depth, 'inference_K': inference_K, 'K': K}



