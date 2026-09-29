from time import perf_counter_ns

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_3.api import DepthAnything3

from depth_estimators.base import BaseDepthEstimator

dav2_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def _prediction_get(prediction, *names):
    for name in names:
        if hasattr(prediction, name):
            return getattr(prediction, name)
        if isinstance(prediction, dict) and name in prediction:
            return prediction[name]
    return None


def _as_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class DepthAnything(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, **kwargs):
        super().__init__(*args, requires_intrinsics=requires_intrinsics, **kwargs)
        self.checkpoint_name = checkpoint_name
        self.version = version

    def load_model(self):
        if self.version == 1:
            raise NotImplementedError
        elif self.version == 2:
            self.model = DepthAnythingV2(**dav2_model_configs[self.checkpoint_name])
            self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.checkpoint_name}.pth', map_location='cpu'))
            self.model.cuda().cuda()
        elif self.version == 3:
            self.model = DepthAnything3.from_pretrained(f"depth-anything/{self.checkpoint_name}").cuda().eval()
        else:
            raise ValueError("Wrong version of DepthAnything")


    @property
    def name(self):
        intrinsics = 'Calib' if self.requires_intrinsics else ''
        return f'DepthAnythingV{self.version}{intrinsics}-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when DepthAnything is used with known focal")

        if self.version == 3:

            input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

            if size is not None:
                input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

            img_h, img_w = input_image.shape[:2]

            input_image = Image.fromarray(input_image)

            if self.requires_intrinsics:
                start_time = perf_counter_ns()
                prediction = self.model.inference([input_image], intrinsics=kwargs['K'][np.newaxis, :, :])
                runtime = perf_counter_ns() - start_time
            else:
                start_time = perf_counter_ns()
                prediction = self.model.inference([input_image])
                runtime = perf_counter_ns() - start_time

            # based on code in depth_anything_v3.utils.io.input_processor
            # there is no cropping and upscaling uses cubic interpolation
            depth = cv2.resize(prediction.depth[0], (img_w, img_h), cv2.INTER_CUBIC)

            return {'depth': depth, 'runtime': runtime}
        elif self.version == 2:
            input_image = cv2.imread(image)

            if size is not None:
                input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

            start_time = perf_counter_ns()
            depth = self.model.infer_image(input_image)
            runtime = perf_counter_ns() - start_time

            return {'depth': 1.0 / depth, 'runtime': runtime}

    def infer_pair(self, image1, image2, size1=None, size2=None, **kwargs):
        if self.version != 3:
            raise NotImplementedError("Pairwise pose regression is only implemented for DepthAnything V3")
        if self.requires_intrinsics and ('K1' not in kwargs.keys() or 'K2' not in kwargs.keys()):
            raise ValueError("Intrinsics K1 and K2 are required for calibrated DepthAnything V3 pair inference")

        input_images = []
        for image, size in ((image1, size1), (image2, size2)):
            input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            if size is not None:
                input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))
            input_images.append(Image.fromarray(input_image))

        start_time = perf_counter_ns()
        if self.requires_intrinsics:
            intrinsics_in = np.stack([kwargs['K1'], kwargs['K2']], axis=0)
            prediction = self.model.inference(input_images, intrinsics=intrinsics_in)
        else:
            prediction = self.model.inference(input_images)
        runtime = perf_counter_ns() - start_time

        depth = _prediction_get(prediction, 'depth', 'depths')
        intrinsics = _prediction_get(prediction, 'intrinsics', 'K')
        extrinsics = _prediction_get(prediction, 'extrinsics', 'extrinsic')
        camera_poses = _prediction_get(prediction, 'camera_poses', 'poses', 'cam2world')

        if depth is None:
            raise ValueError("DepthAnything V3 prediction did not return depth")
        depth = _as_numpy(depth)
        intrinsics = _as_numpy(intrinsics)
        if intrinsics is None:
            intrinsics = np.stack([kwargs.get('K1'), kwargs.get('K2')], axis=0) if self.requires_intrinsics else None
        if extrinsics is None:
            if camera_poses is None:
                raise ValueError("DepthAnything V3 prediction did not return camera poses")
            extrinsics = np.linalg.inv(_as_numpy(camera_poses))
        else:
            extrinsics = _as_numpy(extrinsics)

        return {
            "depth1": depth[0],
            "depth2": depth[1],
            "K1": None if intrinsics is None else np.asarray(intrinsics)[0],
            "K2": None if intrinsics is None else np.asarray(intrinsics)[1],
            "extrinsic1": extrinsics[0],
            "extrinsic2": extrinsics[1],
            "runtime": runtime,
        }




