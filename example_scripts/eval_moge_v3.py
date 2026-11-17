"""Example: evaluating MoGe-3 (https://github.com/microsoft/MoGe) on the D2P benchmark with eval_my_mde.py.

MoGe-3 needs a newer version of the moge package than the one used for the other MDEs in this repository (it requires
numpy>=2 and FlexGEMM), so install it into a separate environment together with the requirements of eval_my_mde.py
(see the README):

    pip install git+https://github.com/microsoft/MoGe.git

FlexGEMM needs a recent PyTorch / Triton (it fails to compile with torch 2.6 / Triton 3.2). We used torch 2.14 built
for CUDA 13.0: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

Run from the root of the repository (accepts the same arguments as eval_my_mde.py):

    python example_scripts/eval_moge_v3.py                # MoGe-3 predicts depth and intrinsics
    python example_scripts/eval_moge_v3.py --use_gt_K     # MoGe-3 gets the GT field of view as input
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


class MoGeV3(BaseDepthEstimator):
    def __init__(self, checkpoint_name='moge-3-vitl', refine_steps=3, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_name = checkpoint_name
        self.refine_steps = refine_steps  # sparse 3D refinement steps, 3 is the MoGe-3 default
        self.model = None

    def load_model(self):
        from moge.model.v3 import MoGeModel
        self.model = MoGeModel.from_pretrained(f'Ruicheng/{self.checkpoint_name}').cuda().eval()
        # the first inference compiles the Triton kernels of the refinement (~20 s), do it here so that it is not
        # included in the runtime of the first image
        with torch.no_grad():
            self.model.infer(torch.rand(3, 480, 640, device='cuda'), refine_steps=self.refine_steps, use_fp16=True)
        torch.cuda.synchronize()

    @property
    def name(self):
        # with --use_gt_K, eval_my_mde.py turns this into MoGeV3Calib-moge-3-vitl
        return f'MoGeV3-{self.checkpoint_name}'

    def infer(self, image, K=None, size=None, **kwargs):
        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))
        input_image = torch.tensor(input_image / 255, dtype=torch.float32).permute(2, 0, 1).to('cuda')
        img_h, img_w = input_image.shape[-2:]

        # MoGe takes the horizontal field of view (in degrees) as an optional input
        fov_x = None if K is None else 2 * np.arctan(img_w / (2 * K[0, 0])) * 180 / np.pi

        start_time = perf_counter_ns()
        with torch.no_grad():
            output = self.model.infer(input_image, fov_x=fov_x, refine_steps=self.refine_steps, use_fp16=True)
        torch.cuda.synchronize()
        runtime = perf_counter_ns() - start_time

        # MoGe returns normalized intrinsics, the benchmark needs them in pixels
        K_pred = output['intrinsics'].cpu().numpy()
        K_pred[0, :] *= img_w
        K_pred[1, :] *= img_h
        # pixels masked out by MoGe have infinite depth, they are ignored by the benchmark
        depth = output['depth'].cpu().numpy()

        return {'depth': depth, 'K': K_pred, 'runtime': runtime}


if __name__ == '__main__':
    main(parse_args(), model=MoGeV3())
