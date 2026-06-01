from __future__ import annotations

import math

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


class MLPPatchSelector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        patch_stride: int,
        top_k: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.top_k = top_k
        self.scorer_select = nn.Sequential(
            nn.Linear(patch_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return patches.transpose(-1, -2).contiguous()

    def _score_candidates(self, patches: torch.Tensor) -> torch.Tensor:
        patch_series = patches.transpose(-1, -2).contiguous()
        channel_scores = self.scorer_select(patch_series).squeeze(-1)
        return channel_scores.mean(dim=-1)

    def _select_topk(self, patches: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patch_num, patch_len, channels = patches.shape
        k = min(self.top_k, patch_num)
        if k == patch_num:
            top_idx = torch.arange(patch_num, device=patches.device).view(1, patch_num).expand(batch, -1)
            top_scores = torch.gather(scores, dim=1, index=top_idx)
        else:
            recent_idx = torch.full((batch, 1), patch_num - 1, dtype=torch.long, device=patches.device)
            candidate_scores = scores[:, :-1]
            candidate_k = k - 1
            top_scores, top_idx = torch.topk(candidate_scores, k=candidate_k, dim=1)
            top_idx = torch.cat([top_idx, recent_idx], dim=1)
            recent_scores = torch.gather(scores, dim=1, index=recent_idx)
            top_scores = torch.cat([top_scores, recent_scores], dim=1)

        order = torch.argsort(top_idx, dim=1)
        top_idx = torch.gather(top_idx, dim=1, index=order)
        top_scores = torch.gather(top_scores, dim=1, index=order)
        gather_idx = top_idx.view(batch, k, 1, 1).expand(-1, -1, patch_len, channels)
        selected = torch.gather(patches, dim=1, index=gather_idx)
        top_weights = torch.sigmoid(top_scores)
        selected = selected * (top_weights / top_weights.detach().clamp_min(1e-6)).view(batch, k, 1, 1)
        return selected, top_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self._patchify(x)
        scores = self._score_candidates(patches)
        selected, selected_indices = self._select_topk(patches, scores)
        return selected, scores, selected_indices


class PatchPassthrough(nn.Module):
    def __init__(self, patch_len: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return patches.transpose(-1, -2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self._patchify(x)
        batch, patch_num, _patch_len, _channels = patches.shape
        scores = x.new_zeros(batch, patch_num)
        indices = torch.arange(patch_num, device=x.device).view(1, patch_num).expand(batch, -1)
        return patches, scores, indices


BidirectionalPatchSelector = MLPPatchSelector


class GatedAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) / math.sqrt(hidden_size))
        self.score = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.score(states))
        logits = torch.matmul(energy, self.query)
        attn = torch.softmax(logits, dim=1)
        context = torch.sum(states * attn.unsqueeze(-1), dim=1)
        gate = torch.sigmoid(self.gate(context))
        context = self.dropout(context * gate)
        return context, attn, gate
