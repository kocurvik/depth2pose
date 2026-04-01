from time import perf_counter_ns

import cv2
import numpy as np
from PIL import Image

from depth_anything_3.api import DepthAnything3

from depth_estimators.base import BaseDepthEstimator


class DepthAnything(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, **kwargs):
        super().__init__(*args, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.version = version

    def load_model(self):
        if self.version == 1:
            raise NotImplementedError
        elif self.version == 2:
            raise NotImplementedError
        elif self.version == 3:
            self.model = DepthAnything3.from_pretrained(f"depth-anything/{self.checkpoint_name}").cuda().eval()
        else:
            raise ValueError("Wrong version of DepthAnything")


    @property
    def name(self):
        intrinsics = 'K' if self.requires_intrinsics else ''
        return f'DepthAnythingV{self.version}{intrinsics}-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when DepthAnything is used with known focal")

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



