from multiprocessing.managers import Value
from time import perf_counter_ns

import cv2
import torch
from unidepth.models import UniDepthV1, UniDepthV2

from unidepth.utils.camera import Pinhole, Fisheye624

from depth_estimators.base import BaseDepthEstimator

class UniDepth(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)

        self.version = version
        self.checkpoint_name = checkpoint_name
        self.requires_intrinsics = requires_intrinsics

    def load_model(self):
        # works without installing unidepth repo
        # self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=f'v{self.version}',
        #                             backbone=self.checkpoint_name, pretrained=True, trust_repo=True).cuda()

        if self.version == 1:
            self.model = UniDepthV1.from_pretrained(f"lpiccinelli/unidepth-v1-{self.checkpoint_name}").cuda()
        elif self.version == 2:
            self.model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{self.checkpoint_name}").cuda()
        else:
            raise ValueError(f"UniDepth available in v1 and v2 only, requested v{self.version}")


    @property
    def name(self):
        return f'UniDepth{self.version}-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when UniDepth is used with known focal")

        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

        input_image = torch.tensor(input_image, dtype=torch.float32, device=self.model.device).permute(2, 0, 1)

        if self.requires_intrinsics and self.version == 2:
            camera = Pinhole(K=torch.tensor(kwargs['K'], dtype=torch.float32, device=self.model.device))
            start_time = perf_counter_ns()
            output = self.model.infer(input_image, camera)
            runtime = perf_counter_ns() - start_time
        elif self.requires_intrinsics and self.version == 1:
            camera = torch.tensor(kwargs['K'], dtype=torch.float32, device=self.model.device)
            start_time = perf_counter_ns()
            output = self.model.infer(input_image, camera)
            runtime = perf_counter_ns() - start_time
        else:
            start_time = perf_counter_ns()
            output = self.model.infer(input_image)
            runtime = perf_counter_ns() - start_time

        depth = output['depth'][0, 0].cpu().numpy()
        K = output['intrinsics'][0].cpu().numpy()

        return {'depth': depth, 'K': K, 'runtime': runtime}



