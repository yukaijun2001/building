from __future__ import annotations

import math

import torch
from torch import nn


class FrequencyDenoiser(nn.Module):
    def __init__(self, seq_len: int, channels: int, keep_ratio: float = 0.35) -> None:
        super().__init__()
        freq_bins = seq_len // 2 + 1
        keep_bins = max(1, int(freq_bins * keep_ratio))
        mask = torch.zeros(freq_bins)
        mask[:keep_bins] = 1.0
        self.register_buffer("low_pass_mask", mask.view(1, freq_bins, 1))
        self.freq_gate = nn.Parameter(torch.ones(1, freq_bins, channels))
        init_alpha = 0.2
        self.mix_logit = nn.Parameter(torch.tensor(math.log(init_alpha / (1.0 - init_alpha))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        gate = torch.sigmoid(self.freq_gate)
        spectrum = spectrum * self.low_pass_mask * gate
        low = torch.fft.irfft(spectrum, n=x.size(1), dim=1)
        alpha = torch.sigmoid(self.mix_logit)
        return x + alpha * (low - x)
