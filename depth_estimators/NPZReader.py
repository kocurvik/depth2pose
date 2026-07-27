import os

import numpy as np

from depth_estimators.base import BaseDepthEstimator


class NPZReader(BaseDepthEstimator):
    def __init__(self,  weights,  **kwargs):
        super().__init__()
        self.weights = weights

    def load_model(self):
        return

    def name(self):
        return "3DGS"

    def infer(self, image, **kwargs):
        gs_depth_path = str(image).replace('images', self.weights).replace('png', 'npz')
        return {'depth': np.load(gs_depth_path)['depth'].astype(np.float32), 'runtime': 0}