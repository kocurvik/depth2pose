from time import perf_counter_ns

import cv2
import torch
from depth_estimators.base import BaseDepthEstimator


class Metric3D(BaseDepthEstimator):
    def __init__(self, checkpoint_name, *args, version=2, requires_intrinsics=False, max_dim=None, **kwargs):
        super().__init__(*args, max_dim=max_dim, requires_intrinsics=requires_intrinsics, **kwargs)

        self.version = version
        self.checkpoint_name = checkpoint_name
        self.requires_intrinsics = requires_intrinsics

    def load_model(self):
        if self.version == 2:
            self.model = torch.hub.load('yvanyin/metric3d', f'metric3d_{self.checkpoint_name}',
                                        pretrain=True).cuda().eval()
        else:
            raise ValueError("Metric3D available only in v2")

    @property
    def name(self):
        return f'Metric3DV{self.version}-{self.checkpoint_name}'

    def infer(self, image, size=None, **kwargs):
        if self.requires_intrinsics and 'K' not in kwargs.keys():
            raise ValueError("Intrinsics are required as input to inference when UniDepth is used with known focal")

        input_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        if size is not None:
            input_image = cv2.resize(input_image, (int(size[0]), int(size[1])))

        input_size = (616, 1064)  # for vit model
        # input_size = (544, 1216) # for convnext model

        h, w = input_image.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(input_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # this is not used since we do not care about metric depth
        # intrinsic = [kwargs['K'][0, 0] * scale, kwargs['K'][1, 1] * scale,
        #              kwargs['K'][0, 2] * scale, kwargs['K'][1, 2] * scale]
        # padding to input_size

        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                 cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        ###################### canonical camera space ######################
        # inference
        start_time = perf_counter_ns()
        pred_depth, confidence, output_dict = self.model.inference({'input': rgb})
        runtime = perf_counter_ns() - start_time

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                     pad_info[2]: pred_depth.shape[1] - pad_info[3]]

        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], input_image.shape[:2],
                                                     mode='bilinear').squeeze()

        # only if we care about metric depth
        # canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
        # pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
        # pred_depth = torch.clamp(pred_depth, 0, 300)

        return {'depth': pred_depth.cpu().numpy(), 'runtime': runtime}
