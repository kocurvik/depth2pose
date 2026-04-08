from time import perf_counter_ns

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_3.api import DepthAnything3

from depth_estimators.base import BaseDepthEstimator

dav2_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class DepthAnything(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, **kwargs):
        super().__init__(*args, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.version = version

    def load_model(self):
        if self.version == 1:
            raise NotImplementedError
        elif self.version == 2:
            self.model = DepthAnythingV2(**dav2_model_configs[self.checkpoint_name])
            self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.checkpoint_name}.pth', map_location='cpu'))
            self.model.cuda().cuda()
        elif self.version == 3:
            self.model = DepthAnything3.from_pretrained(f"depth-anything/{self.checkpoint_name}").cuda().eval()
        else:
            raise ValueError("Wrong version of DepthAnything")


    @property
    def name(self):
        intrinsics = 'Calib' if self.requires_intrinsics else ''
        return f'DepthAnythingV{self.version}{intrinsics}-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when DepthAnything is used with known focal")

        if self.version == 3:

            input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

            if size is not None:
                input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

            img_h, img_w = input_image.shape[:2]

            input_image = Image.fromarray(input_image)

            if self.requires_intrinsics:
                start_time = perf_counter_ns()
                prediction = self.model.inference([input_image], intrinsics=kwargs['K'][np.newaxis, :, :])
                runtime = perf_counter_ns() - start_time
            else:
                start_time = perf_counter_ns()
                prediction = self.model.inference([input_image])
                runtime = perf_counter_ns() - start_time

            # based on code in depth_anything_v3.utils.io.input_processor
            # there is no cropping and upscaling uses cubic interpolation
            depth = cv2.resize(prediction.depth[0], (img_w, img_h), cv2.INTER_CUBIC)

            return {'depth': depth, 'runtime': runtime}
        elif self.version == 2:
            input_image = cv2.imread(image)

            if size is not None:
                input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

            start_time = perf_counter_ns()
            depth = self.model.infer_image(input_image)
            runtime = perf_counter_ns() - start_time

            return {'depth': 1.0 / depth, 'runtime': runtime}




