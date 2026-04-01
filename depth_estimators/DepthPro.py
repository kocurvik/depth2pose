from time import perf_counter_ns

import cv2
import depth_pro
import numpy as np
import torch
from depth_estimators.base import BaseDepthEstimator

class DepthPro(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=1, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)

        self.version = version
        self.checkpoint_name = checkpoint_name
        self.requires_intrinsics = requires_intrinsics

    def load_model(self):
        if self.version == 1:
            torch.backends.cudnn.benchmark = True  # Pick fastest kernels
            torch.set_float32_matmul_precision('high')
            model, self.transform = depth_pro.create_model_and_transforms(device=torch.device('cuda'))
            self.model = torch.compile(model, mode="reduce-overhead").eval()

        else:
            raise ValueError("DepthPro available only in v1")

    @property
    def name(self):
        if self.requires_intrinsics:
            return f'DepthProCalib-{self.checkpoint_name}'
        return f'DepthPro-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):


        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when UniDepth is used with known focal")

        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        if size is not None:
            image = cv2.resize(image, (int(size[0]), int(size[1])))

        if self.requires_intrinsics:
            f_px = torch.tensor((kwargs['K'][0, 0] + kwargs['K'][1, 1]) / 2).cuda()
        else:
            f_px = None


        with torch.autocast('cuda'):
            transformed_image = self.transform(image).cuda()
            start_time = perf_counter_ns()
            prediction = self.model.infer(transformed_image, f_px=f_px)
            runtime = perf_counter_ns() - start_time

        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"].cpu().numpy()[()]
        K = np.array([[focallength_px, 0, image.shape[1] / 2], [0, focallength_px, image.shape[0] / 2], [0,0,1]])

        return {'depth': depth.cpu().numpy(), 'K': K, 'runtime': runtime}
