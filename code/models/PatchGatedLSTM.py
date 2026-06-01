from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from layers import FrequencyDenoiser, GatedAttention, MLPPatchSelector, RevIN


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    patch_scores: torch.Tensor
    selected_indices: torch.Tensor
    attention: torch.Tensor


class PatchGatedLSTMForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        target_index: int,
        patch_len: int,
        patch_stride: int,
        top_k_patches: int,
        hidden_size: int,
        model_dim: int,
        lstm_layers: int,
        dropout: float,
        fft_keep_ratio: float,
    ) -> None:
        super().__init__()
        self.target_index = target_index
        self.revin = RevIN(input_dim)
        self.nlinear = nn.Linear(seq_len, pred_len)
        self.denoiser = FrequencyDenoiser(seq_len, input_dim, fft_keep_ratio)
        self.patch_selector = MLPPatchSelector(
            input_dim=input_dim,
            patch_len=patch_len,
            patch_stride=patch_stride,
            top_k=top_k_patches,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.patch_projection = nn.Linear(input_dim, model_dim)
        self.encoder = nn.LSTM(
            model_dim,
            hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.gated_attention = GatedAttention(hidden_size, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x_norm, mean, std = self.revin.norm(x)
        target_seq = x_norm[:, :, self.target_index]
        last = target_seq[:, -1:]
        baseline = self.nlinear(target_seq - last) + last
        x_denoised = self.denoiser(x_norm)
        selected, patch_scores, selected_indices = self.patch_selector(x_denoised)

        batch, top_k, patch_len, channels = selected.shape
        tokens = selected.reshape(batch, top_k * patch_len, channels)
        tokens = self.patch_projection(tokens)

        states, _ = self.encoder(tokens)
        context, attention, _gate = self.gated_attention(states)
        residual = self.head(context)
        pred = (baseline + residual).unsqueeze(-1)
        pred = self.revin.denorm(pred, mean, std, self.target_index)
        return ModelOutput(pred, patch_scores, selected_indices, attention)


class SegmentLSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, pred_len: int, seg_len: int, dropout: float) -> None:
        super().__init__()
        if pred_len % seg_len != 0:
            raise ValueError(f"pred_len={pred_len} must be divisible by seg_len={seg_len}")
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.seg_num_y = pred_len // seg_len
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, hidden_size))
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, seg_len),
        )

    def forward(self, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.size(1)
        query = self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        out, _ = self.decoder(query, (hidden[-1:].contiguous(), cell[-1:].contiguous()))
        return self.predict(out).reshape(batch_size, self.pred_len)


class PatchGatedSegRNNForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        target_index: int,
        patch_len: int,
        patch_stride: int,
        top_k_patches: int,
        hidden_size: int,
        model_dim: int,
        lstm_layers: int,
        dropout: float,
        fft_keep_ratio: float,
        seg_len: int,
    ) -> None:
        super().__init__()
        self.target_index = target_index
        self.revin = RevIN(input_dim)
        self.nlinear = nn.Linear(seq_len, pred_len)
        self.denoiser = FrequencyDenoiser(seq_len, input_dim, fft_keep_ratio)
        self.patch_selector = MLPPatchSelector(
            input_dim=input_dim,
            patch_len=patch_len,
            patch_stride=patch_stride,
            top_k=top_k_patches,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.patch_projection = nn.Linear(input_dim, model_dim)
        self.encoder = nn.LSTM(
            model_dim,
            hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.seg_decoder = SegmentLSTMDecoder(hidden_size, pred_len, seg_len, dropout)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x_norm, mean, std = self.revin.norm(x)
        target_seq = x_norm[:, :, self.target_index]
        last = target_seq[:, -1:]
        baseline = self.nlinear(target_seq - last) + last
        x_denoised = self.denoiser(x_norm)
        selected, patch_scores, selected_indices = self.patch_selector(x_denoised)

        batch, top_k, patch_len, channels = selected.shape
        tokens = selected.reshape(batch, top_k * patch_len, channels)
        tokens = self.patch_projection(tokens)

        _states, (hidden, cell) = self.encoder(tokens)
        residual = self.seg_decoder(hidden, cell)
        pred = (baseline + residual).unsqueeze(-1)
        pred = self.revin.denorm(pred, mean, std, self.target_index)
        attention = x.new_empty(batch, 0)
        return ModelOutput(pred, patch_scores, selected_indices, attention)
