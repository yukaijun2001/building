from __future__ import annotations

import torch
from torch import nn


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


BidirectionalPatchSelector = MLPPatchSelector
