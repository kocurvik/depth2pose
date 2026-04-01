from time import perf_counter_ns

import cv2
import torch

from unik3d import UniK3D as UniK3DImpl
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)

from depth_estimators.base import BaseDepthEstimator

class UniK3D(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)

        self.version = version
        self.checkpoint_name = checkpoint_name
        self.requires_intrinsics = requires_intrinsics

    def load_model(self):
        if self.version == 1:
            self.model = UniK3DImpl.from_pretrained(f"lpiccinelli/unik3d-{self.checkpoint_name}").cuda().eval()
        else:
            raise ValueError(f"UniK3D available in v1, requested v{self.version}")


    @property
    def name(self):
        if self.requires_intrinsics:
            return f'UniK3DCalib-{self.checkpoint_name}'
        return f'UniKD3-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when UniDepth is used with known focal")

        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

        input_image = torch.tensor(input_image, dtype=torch.float32, device=self.model.device).permute(2, 0, 1)

        if self.requires_intrinsics:
            camera = Pinhole(K=torch.tensor(kwargs['K'], dtype=torch.float32, device=self.model.device))
            start_time = perf_counter_ns()
            output = self.model.infer(input_image, camera)
            runtime = perf_counter_ns() - start_time
        else:
            start_time = perf_counter_ns()
            output = self.model.infer(input_image)
            runtime = perf_counter_ns() - start_time

        depth = output['depth'][0, 0].cpu().numpy()
        # K = output['intrinsics'][0].cpu().numpy()
        # TODO once published on Github implement K from rays

        return {'depth': depth, 'runtime': runtime}



