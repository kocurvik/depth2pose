"""Example: evaluating SurGe (https://github.com/karimknaebel/surge) on the D2P benchmark with eval_my_mde.py.

SurGe needs python>=3.12 and torch>=2.12, so install it into a separate environment together with the requirements of
eval_my_mde.py (see the README):

    conda create -n d2p_surge python=3.12
    conda activate d2p_surge
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    pip install git+https://github.com/karimknaebel/surge.git

Note that the SurGe weights are licensed under CC BY-NC 4.0.

Run from the root of the repository (accepts the same arguments as eval_my_mde.py):

    python example_scripts/eval_surge.py                # SurGe predicts depth and intrinsics
    python example_scripts/eval_surge.py --use_gt_K     # SurGe gets the GT field of view as input
"""
import os
import sys
from time import perf_counter_ns

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimators.base import BaseDepthEstimator
from eval_my_mde import main, parse_args


class SurGe(BaseDepthEstimator):
    def __init__(self, checkpoint_name='surge-large', num_tokens='max', **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_name = checkpoint_name
        # ViT token budget, 'max' (2802) is the SurGe default, 'min' (1024) is faster but less accurate
        self.num_tokens = num_tokens
        self.model = None

    def load_model(self):
        from surge import SurGe as SurGeModel
        self.model = SurGeModel.from_pretrained(f'karimknaebel/{self.checkpoint_name}').cuda().eval()
        # the first inference compiles the Triton kernels of the neighborhood attention, do it here so that it is not
        # included in the runtime of the first image
        with torch.autocast('cuda', dtype=torch.float16):
            self.model.infer(torch.rand(1, 3, 480, 640, device='cuda'), num_tokens=self.num_tokens)
        torch.cuda.synchronize()

    @property
    def name(self):
        # with --use_gt_K, eval_my_mde.py turns this into SurGeCalib-surge-large
        return f'SurGe-{self.checkpoint_name}'

    def infer(self, image, K=None, size=None, **kwargs):
        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))
        # SurGe expects BCHW RGB images in [0, 1] without any normalization
        input_image = torch.tensor(input_image / 255, dtype=torch.float32).permute(2, 0, 1)[None].to('cuda')
        img_h, img_w = input_image.shape[-2:]

        # SurGe takes the horizontal field of view (in degrees) as an optional input
        fov_x = None if K is None else 2 * np.arctan(img_w / (2 * K[0, 0])) * 180 / np.pi

        start_time = perf_counter_ns()
        with torch.autocast('cuda', dtype=torch.float16):
            output = self.model.infer(input_image, num_tokens=self.num_tokens, fov_x=fov_x)
        torch.cuda.synchronize()
        runtime = perf_counter_ns() - start_time

        # SurGe returns normalized intrinsics, the benchmark needs them in pixels
        K_pred = output['intrinsics'][0].float().cpu().numpy()
        K_pred[0, :] *= img_w
        K_pred[1, :] *= img_h
        # depth is known only up to scale, invalid values (<= 0, nan, inf) are ignored by the benchmark
        depth = output['depth'][0].float().cpu().numpy()

        return {'depth': depth, 'K': K_pred, 'runtime': runtime}


if __name__ == '__main__':
    main(parse_args(), model=SurGe())
