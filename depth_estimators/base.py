import cv2
import torch

class BaseDepthEstimator(torch.nn.Module):
    def __init__(self, *args, max_dim=None, requires_intrinsics=False,  **kwargs):
        super().__init__()
        self.requires_intrinsics = requires_intrinsics
        self.max_dim = max_dim

    def load_model(self):
        raise NotImplementedError

    def name(self):
        return "BaseDepthEstimator"

    def infer(self, image, **kwargs):
        raise NotImplementedError

    def infer_pair(self, image1, image2, **kwargs):
        raise NotImplementedError





