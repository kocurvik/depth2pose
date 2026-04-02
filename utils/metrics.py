import torch
from torch import Tensor
from torch.nn import functional as F
from functools import partial
from collections import defaultdict





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
        return (inlier < 1.25**exponent).to(torch.float32).mean() * 100

    @staticmethod
    def tau(tensor1, tensor2, perc):
        inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
        return (inlier < (1.0 + perc)).to(torch.float32).mean()

    @staticmethod
    def ssi(tensor1, tensor2):
        stability_mat = 1e-9 * torch.eye(2, device=tensor1.device)
        tensor2_one = torch.stack([tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1)
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
    def arel(tensor1, tensor2):
        return (torch.abs(tensor1 - tensor2) / tensor1).mean()
    
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
        deltas = [DepthMetrics.delta(tensor1, tensor2, exponent) for exponent in exponents]
        return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0
    
    def evaluate(self, gts: Tensor, pds: Tensor, masks: Tensor, max_depth: float = None):
        results = defaultdict(list)
        pds = F.interpolate(pds, gts.shape[-2:], mode='bilinear')

        for i, (gt, pd, mask) in enumerate(zip(gts, pds, masks)):
            if max_depth is not None:
                mask &= (gt <= max_depth)
            for metric_name in self.metrics_to_use:
                for rescale_fn in ['si', 'ssi']: # scale-invariant and affine-invariant
                    results[f"{metric_name}_{rescale_fn}"].append(self.metrics[metric_name](gt[mask], eval(f"DepthMetrics.{rescale_fn}")(gt[mask], pd[mask])).mean())
                results[metric_name].append(self.metrics[metric_name](gt[mask], pd[mask]).mean())
        return {name: torch.stack(vals, dim=0).mean().item() for name, vals in results.items()}



if __name__ == '__main__':
    gts = torch.randn(10, 1, 224, 224) * 10
    pds = torch.randn(10, 1, 224, 224) * 10
    masks = torch.ones(10, 1, 224, 224).bool()
    metrics = DepthMetrics()
    results = metrics.evaluate(gts, pds, masks)
    print(results)