import cv2
import torch

class BaseDepthEstimator(torch.nn.Module):
    def __init__(self, *args, max_dim=None, requires_intrinsics=False,  **kwargs):
        super().__init__()
        self.requires_intrinsics = requires_intrinsics
        self.max_dim = max_dim

    def name(self):
        return "BaseDepthEstimator"

    def enforce_max_dim(self, image):
        if self.max_dim is None:
            return image, 1.0

        height, width = image.shape[0], image.shape[1]

        if self.max_dim < max(width, height):
            return image, 1.0

        scale = self.max_dim / max(width, height)

        return cv2.resize(image, None, fx=scale, fy=scale), scale

    def infer(self, image, **kwargs):
        raise NotImplementedError





