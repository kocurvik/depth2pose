import os

import numpy as np

from depth_estimators.base import BaseDepthEstimator


class NPZReader(BaseDepthEstimator):
    def __init__(self,   **kwargs):
        super().__init__()

    def load_model(self):
        return

    def name(self):
        return "3DGS"

    def infer(self, image, **kwargs):
        gs_depth_path = str(image).replace('images', 'gs_depths').replace('png', 'npz')
        return {'depth': np.load(gs_depth_path)['depth'].astype(np.float32), 'runtime': 0}