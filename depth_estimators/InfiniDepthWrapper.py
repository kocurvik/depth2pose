from time import perf_counter_ns

import cv2
import numpy as np
import torch
from InfiniDepth.utils.inference_utils import resolve_output_size_from_mode

from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS
from InfiniDepth.utils.model_utils import build_model

from depth_estimators.base import BaseDepthEstimator


class InfiniDepth(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, requires_intrinsics=False, **kwargs):
        super().__init__(*args, requires_intrinsics=requires_intrinsics, **kwargs)

        if checkpoint_name != 'vitl':
            raise ValueError("InfiniDepth only available in vitl version!")
        self.checkpoint_name = checkpoint_name
        if requires_intrinsics:
            raise ValueError("InfiniDepth doesnt not have a mode with included intrinsics for depth prediction!")
        self.requires_intrinsics = False

    def load_model(self):
        self.model = build_model("InfiniDepth", model_path='checkpoints/infinidepth.ckpt')

    @property
    def name(self):
        return f'InfiniDepth-{self.checkpoint_name}'

    def enforce_inference_size(self, orig_w, orig_h):
        return np.round(orig_w / 16).astype(int) * 16, np.round(orig_h / 16).astype(int) * 16


    def infer(self, image, size=None, **kwargs):
        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)

        orig_h, orig_w = input_image.shape[:2]

        inference_w, inference_h = self.enforce_inference_size(orig_w, orig_h)

        input_image = cv2.resize(input_image, (inference_w, inference_h), interpolation=cv2.INTER_AREA)

        input_image = torch.tensor(input_image, dtype=torch.float32, device='cuda').permute(2, 0, 1) / 255

        h_sample, w_sample = resolve_output_size_from_mode(
            output_resolution_mode='original',
            org_h=orig_h,
            org_w=orig_w,
            h=inference_h,
            w=inference_w,
            output_size=(orig_h, orig_w),
            upsample_ratio=1.0,
        )

        start_time = perf_counter_ns()
        query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"]((h_sample, w_sample)).unsqueeze(0).to('cuda')
        pred_2d_uniform_depth, _ = self.model.inference(
            image=input_image.unsqueeze(0),
            query_coord=query_2d_uniform_coord,
        )
        pred_depthmap = pred_2d_uniform_depth.permute(0, 2, 1).view(1, 1, h_sample, w_sample)
        runtime = perf_counter_ns() - start_time

        depth = pred_depthmap[0, 0].cpu().numpy()

        return {'depth': depth, 'runtime': runtime}



