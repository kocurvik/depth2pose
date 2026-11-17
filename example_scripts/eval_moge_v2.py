"""Example: evaluating MoGe v2 (https://github.com/microsoft/MoGe) on the D2P benchmark with eval_my_mde.py.

This file shows a complete implementation of the MyMDE class from eval_my_mde.py for MoGe v2. To evaluate your own MDE
either implement MyMDE directly in eval_my_mde.py or write a script like this one.

Run from the root of the repository (accepts the same arguments as eval_my_mde.py):

    python example_scripts/eval_moge_v2.py                # MoGe v2 predicts depth and intrinsics
    python example_scripts/eval_moge_v2.py --use_gt_K     # MoGe v2 gets the GT field of view as input

MoGe v2 is already in the results table (as MoGeV2-moge-2-vitl / MoGeV2Calib-moge-2-vitl), so the script stops before
running anything. To reproduce its results without modifying the tracked CSV files, write them elsewhere, e.g.
--out_dir /tmp/d2p_results (or replace the rows in csv_results with --overwrite).
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


class MoGeV2(BaseDepthEstimator):
    def __init__(self, checkpoint_name='moge-2-vitl', **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_name = checkpoint_name
        self.model = None

    def load_model(self):
        from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(f'Ruicheng/{self.checkpoint_name}').cuda().eval()

    @property
    def name(self):
        # with --use_gt_K, eval_my_mde.py turns this into MoGeV2Calib-moge-2-vitl
        return f'MoGeV2-{self.checkpoint_name}'

    def infer(self, image, K=None, size=None, **kwargs):
        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))
        input_image = torch.tensor(input_image / 255, dtype=torch.float32).permute(2, 0, 1).to('cuda')
        img_h, img_w = input_image.shape[-2:]

        # MoGe takes the horizontal field of view (in degrees) as an optional input
        fov_x = None if K is None else 2 * np.arctan(img_w / (2 * K[0, 0])) * 180 / np.pi

        start_time = perf_counter_ns()
        output = self.model.infer(input_image[None], fov_x=fov_x, use_fp16=True)
        runtime = perf_counter_ns() - start_time

        # MoGe returns normalized intrinsics, the benchmark needs them in pixels
        K_pred = output['intrinsics'][0].cpu().numpy()
        K_pred[0, :] *= img_w
        K_pred[1, :] *= img_h
        depth = output['depth'][0].cpu().numpy()

        return {'depth': depth, 'K': K_pred, 'runtime': runtime}


if __name__ == '__main__':
    main(parse_args(), model=MoGeV2())
