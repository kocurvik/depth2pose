import math
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def get_nested_dict(d, keys, default=None):
    for k in keys:
        d = d.get(k, default)
        if d is None:
            break
    return d


def set_nested_dict(d, keys, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def traverse_nested_dict_keys(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_key in traverse_nested_dict_keys(v):
                yield (k,) + sub_key
        else:
            yield (k,)


def key_average(list_of_dicts: list) -> dict[str]:
    """
    Returns a dictionary with the average value of each key in the input list of dictionaries.
    """
    _nested_dict_keys = set()
    for d in list_of_dicts:
        _nested_dict_keys.update(traverse_nested_dict_keys(d))
    _nested_dict_keys = sorted(_nested_dict_keys)
    result = {}
    for k in _nested_dict_keys:
        values = []
        for d in list_of_dicts:
            v = get_nested_dict(d, k)
            if v is not None and not math.isnan(v):
                values.append(v)
        avg = sum(values) / len(values) if values else float("nan")
        set_nested_dict(result, k, avg)
    return result


class DepthMetrics:
    def __init__(self):
        self.metrics = {
            "d1": partial(DepthMetrics.delta, exponent=1.0),
            "d2": partial(DepthMetrics.delta, exponent=2.0),
            "d3": partial(DepthMetrics.delta, exponent=3.0),
            "tau": partial(DepthMetrics.tau, perc=0.03),
            "A.Rel": DepthMetrics.arel,
            "RMS": DepthMetrics.rmse,
            "SI_log": DepthMetrics.silog,
            "RMS_log": DepthMetrics.rmse_log,
            "Sq.Rel": DepthMetrics.sqrel,
            "log10": DepthMetrics.log10,
            "medianlog": DepthMetrics.medianlog,
            "d_auc": DepthMetrics.d_auc,
        }
        self.metrics_to_use = ["A.Rel", "d1"]

    @staticmethod
    def rmse(tensor1, tensor2):
        return torch.sqrt(((tensor1 - tensor2) ** 2).mean())

    @staticmethod
    def rmse_log(tensor1, tensor2):
        return torch.sqrt(((torch.log(tensor1) - torch.log(tensor2)) ** 2).mean())

    @staticmethod
    def delta(tensor1, tensor2, exponent):
        inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
        return (inlier < 1.25**exponent).to(torch.float32).mean().item() * 100

    @staticmethod
    def tau(tensor1, tensor2, perc):
        inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
        return (inlier < (1.0 + perc)).to(torch.float32).mean()

    @staticmethod
    def ssi(tensor1, tensor2):
        stability_mat = 1e-9 * torch.eye(2, device=tensor1.device)
        tensor2_one = torch.stack(
            [tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1
        )
        scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (
            tensor2_one.T @ tensor1.unsqueeze(1)
        )
        scale, shift = scale_shift.squeeze().chunk(2, dim=0)
        return tensor2 * scale + shift

    @staticmethod
    def si(tensor1, tensor2):
        return tensor2 * torch.median(tensor1) / torch.median(tensor2)

    @staticmethod
    def sqrel(tensor1, tensor2):
        return (((tensor1 - tensor2) ** 2) / tensor1).mean()

    @staticmethod
    def arel(tensor1, tensor2, eps: float = 1e-6):
        return (torch.abs(tensor1 - tensor2) / (tensor1 + eps)).mean().item() * 100

    @staticmethod
    def log10(tensor1, tensor2):
        return torch.abs(torch.log10(tensor1) - torch.log10(tensor2)).mean()

    @staticmethod
    def silog(tensor1, tensor2):
        return 100 * torch.std(torch.log(tensor1) - torch.log(tensor2)).mean()

    @staticmethod
    def medianlog(tensor1, tensor2):
        return 100 * (torch.log(tensor1) - torch.log(tensor2)).median().abs()

    @staticmethod
    def d_auc(tensor1, tensor2):
        exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
        deltas = [
            DepthMetrics.delta(tensor1, tensor2, exponent) for exponent in exponents
        ]
        return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0

    def compute_metric_depth(
        self, gt: Tensor, pd: Tensor, mask: Tensor, max_depth: float = None
    ):
        pd_mask = ~torch.isinf(pd)
        mask &= pd_mask
        # print(gt[mask].mean(), pd[mask].mean())
        results = {}
        for metric_name in self.metrics_to_use:
            results[metric_name] = self.metrics[metric_name](gt[mask], pd[mask])
        return results

    def compute_scale_inv_depth(
        self, gt: Tensor, pd: Tensor, mask: Tensor, max_depth: float = None
    ):
        results = {}
        # lr_mask, lr_index = utils3d.pt.masked_nearest_resize(mask=mask, size=(64, 64), return_index=True)
        # pd_depth_lr_masked, gt_depth_lr_masked = pd[lr_index][lr_mask], gt[lr_index][lr_mask]
        # scale, shift = align_depth_scale(pd_depth_lr_masked, gt_depth_lr_masked, 1/gt_depth_lr_masked)
        # pd = pd * scale
        pd_mask = ~torch.isinf(pd)
        mask &= pd_mask
        for metric_name in self.metrics_to_use:
            results[f"{metric_name}_si"] = self.metrics[metric_name](
                gt[mask], DepthMetrics.si(gt[mask], pd[mask])
            )
        return results

    def compute_affine_inv_depth(
        self, gt: Tensor, pd: Tensor, mask: Tensor, max_depth: float = None
    ):
        results = {}
        pd_mask = ~torch.isinf(pd)
        mask &= pd_mask
        for metric_name in self.metrics_to_use:
            results[f"{metric_name}_ssi"] = self.metrics[metric_name](
                gt[mask], DepthMetrics.ssi(gt[mask], pd[mask])
            )
        return results

    def evaluate_all(
        self, gts: Tensor, pds: Tensor, masks: Tensor, max_depth: float = None
    ):
        results = defaultdict(list)
        pds = F.interpolate(pds, gts.shape[-2:], mode="bilinear")

        for i, (gt, pd, mask) in enumerate(zip(gts, pds, masks)):
            if max_depth is not None:
                mask &= gt <= max_depth
            for metric_name in self.metrics_to_use:
                for rescale_fn in ["si", "ssi"]:  # scale-invariant and affine-invariant
                    results[f"{metric_name}_{rescale_fn}"].append(
                        self.metrics[metric_name](
                            gt[mask],
                            eval(f"DepthMetrics.{rescale_fn}")(gt[mask], pd[mask]),
                        ).mean()
                    )
                results[metric_name].append(
                    self.metrics[metric_name](gt[mask], pd[mask]).mean()
                )
        return {
            name: torch.stack(vals, dim=0).mean().item()
            for name, vals in results.items()
        }


if __name__ == "__main__":
    gts = torch.randn(10, 1, 224, 224) * 10
    pds = torch.randn(10, 1, 224, 224) * 10
    masks = torch.ones(10, 1, 224, 224).bool()
    metrics = DepthMetrics()
    results = metrics.evaluate(gts, pds, masks)
    print(results)
