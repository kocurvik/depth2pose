"""Example: evaluating VGGT-Omega (https://github.com/facebookresearch/vggt-omega) on the D2P benchmark with
eval_my_mde.py.

VGGT-Omega needs numpy<2, so install it into a separate environment together with the requirements of eval_my_mde.py
(see the README):

    conda create -n d2p_vggt_omega python=3.12
    conda activate d2p_vggt_omega
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install "numpy<2" git+https://github.com/facebookresearch/vggt-omega.git

The checkpoints are gated: request access at https://huggingface.co/facebook/VGGT-Omega and log in with
`hf auth login` before the first run. We use vggt_omega_1b_416_reproduce.pt, which the authors recommend for
benchmarking (see reproduction.md in their repository). The code and weights are released under the FAIR Noncommercial
Research License.

VGGT-Omega takes no camera intrinsics as input, so run it without --use_gt_K. It predicts the intrinsics itself and
the poses are estimated with both the GT intrinsics (calib) and the predicted ones (mdecalib).

Run from the root of the repository (accepts the same arguments as eval_my_mde.py):

    python example_scripts/eval_vggt_omega.py
"""
import os
import sys
from time import perf_counter_ns

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimators.base import BaseDepthEstimator
from eval_my_mde import main, parse_args


class VGGTOmega(BaseDepthEstimator):
    def __init__(self, checkpoint_file='vggt_omega_1b_416_reproduce.pt', image_resolution=416, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_file = checkpoint_file
        # the 416 checkpoint was trained with a pixel budget of 416 x 416, the input images are resized accordingly
        self.image_resolution = image_resolution
        self.model = None

    def load_model(self):
        from huggingface_hub import hf_hub_download
        from vggt_omega.models import VGGTOmega as VGGTOmegaModel
        checkpoint_path = hf_hub_download('facebook/VGGT-Omega', self.checkpoint_file)
        # the model has ~1.2B parameters (4.6 GB in fp32): create it directly on the GPU and memory-map the checkpoint
        # so that it does not need two copies of the weights in RAM
        with torch.device('cuda'):
            self.model = VGGTOmegaModel().eval()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', mmap=True))
        self.model = self.model.cuda()  # moves the buffers that the model creates explicitly on the CPU
        # the first inference with a new input shape takes ~15 s (CUDA kernel selection), see infer
        self.warmed_up_shapes = set()

    @property
    def name(self):
        return 'VGGTOmega-1b-416'

    def infer(self, image, K=None, size=None, **kwargs):
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if size is not None:
            raise NotImplementedError('resizing the input is not supported in this example')
        with Image.open(image) as img:
            img = img.convert('RGB')
        width, height = img.size

        # Same as load_and_preprocess_images(mode='balanced') of VGGT-Omega: about (image_resolution / 16)^2 tokens
        # with the aspect ratio of the image. Unlike it, images with an aspect ratio outside [0.5, 2] are not
        # center-cropped (D2P has some 1592x715 images), so that the depth covers the whole image.
        num_tokens = (self.image_resolution // 16) ** 2
        w_patches = max(1, int(np.round(np.sqrt(num_tokens * width / height))))
        h_patches = max(1, int(np.round(num_tokens / np.sqrt(num_tokens * width / height))))
        input_h, input_w = 16 * h_patches, 16 * w_patches
        img = img.resize((input_w, input_h), Image.Resampling.BICUBIC)
        input_image = TF.to_tensor(img)[None].to('cuda')

        # untimed warm-up for every new input shape (D2P has only a few), so that the one-off kernel selection is not
        # included in the runtime
        if (input_h, input_w) not in self.warmed_up_shapes:
            with torch.inference_mode():
                self.model(input_image)
            torch.cuda.synchronize()
            self.warmed_up_shapes.add((input_h, input_w))

        start_time = perf_counter_ns()
        with torch.inference_mode():
            predictions = self.model(input_image)
            _, intrinsics = encoding_to_camera(predictions['pose_enc'], (input_h, input_w))
        torch.cuda.synchronize()
        runtime = perf_counter_ns() - start_time

        # depth is (B, S, h, w, 1), resize it to the original image
        depth = predictions['depth'][0, 0, ..., 0].float()
        depth = F.interpolate(depth[None, None], size=(height, width), mode='bilinear', align_corners=False)[0, 0]

        # intrinsics are in pixels of the network input, convert them to the original image
        K_pred = intrinsics[0, 0].float().cpu().numpy()
        K_pred[0, 0] *= width / input_w
        K_pred[1, 1] *= height / input_h
        K_pred[0, 2] = width / 2
        K_pred[1, 2] = height / 2

        return {'depth': depth.cpu().numpy(), 'K': K_pred, 'runtime': runtime}


if __name__ == '__main__':
    args = parse_args()
    if args.use_gt_K:
        sys.exit('VGGT-Omega takes no intrinsics as input, run this script without --use_gt_K.')
    main(args, model=VGGTOmega())
