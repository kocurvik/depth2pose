import cv2
import torch

from depth_estimators.base import BaseDepthEstimator

class UniDepth(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)

        self.version = version
        self.checkpoint_name = checkpoint_name

        self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=f'v{self.version}',
                                    backbone=self.checkpoint_name, pretrained=True, trust_repo=True)

    @property
    def name(self):
        return f'UniDepth{self.version}-{self.checkpoint_name}'

    def infer(self, image, **kwargs):
        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image, dtype=torch.float32, device=self.model.device).permute(2, 0, 1)


        output = self.model.infer(input_image)

        depth = output['depth'][0, 0].cpu().numpy()
        K = output['intrinsics'][0].cpu().numpy()

        return {'depth': depth, 'K': K}



