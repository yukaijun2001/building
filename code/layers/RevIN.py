from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(self.eps)
        z = (x - mean) / std
        if self.affine:
            z = z * self.weight + self.bias
        return z, mean, std

    def denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, target_index: int) -> torch.Tensor:
        if self.affine:
            weight = self.weight[:, :, target_index : target_index + 1].clamp_min(self.eps)
            bias = self.bias[:, :, target_index : target_index + 1]
            x = (x - bias) / weight
        return x * std[:, :, target_index : target_index + 1] + mean[:, :, target_index : target_index + 1]
