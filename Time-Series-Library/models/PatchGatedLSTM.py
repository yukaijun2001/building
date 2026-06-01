from __future__ import annotations

import torch
from torch import nn

from layers.PatchGatedLSTM_layers import (
    FrequencyDenoiser,
    GatedAttention,
    MLPPatchSelector,
    PatchPassthrough,
    RevIN,
)


ABLATION_ALIASES = {
    "none": "none",
    "baseline": "none",
    "full": "none",
    "freq": "no_frequency_denoising",
    "frequency": "no_frequency_denoising",
    "denoising": "no_frequency_denoising",
    "frequency_denoising": "no_frequency_denoising",
    "no_freq": "no_frequency_denoising",
    "no_frequency": "no_frequency_denoising",
    "no_denoising": "no_frequency_denoising",
    "no_frequency_denoising": "no_frequency_denoising",
    "patch": "no_patch_selection",
    "patch_selection": "no_patch_selection",
    "patch_scoring": "no_patch_selection",
    "bidirectional_patch": "no_patch_selection",
    "bidirectional_patch_selection": "no_patch_selection",
    "no_patch": "no_patch_selection",
    "no_patch_scoring": "no_patch_selection",
    "no_patch_selection": "no_patch_selection",
    "gate": "no_gated_attention",
    "gated": "no_gated_attention",
    "attention": "no_gated_attention",
    "gated_attention": "no_gated_attention",
    "no_gate": "no_gated_attention",
    "no_attention": "no_gated_attention",
    "no_gated_attention": "no_gated_attention",
}


def normalize_ablation(value: str) -> str:
    key = str(value or "none").strip().lower().replace("-", "_")
    if key not in ABLATION_ALIASES:
        choices = ", ".join(sorted({"none", "no_frequency_denoising", "no_patch_selection", "no_gated_attention"}))
        raise ValueError(f"Unknown PatchGatedLSTM ablation '{value}'. Choose one of: {choices}")
    return ABLATION_ALIASES[key]


class Model(nn.Module):
    """
    Source-experiment PatchGatedLSTM adapted to the Time-Series-Library model API.

    The forecasting path intentionally keeps the original target-only NLinear
    baseline with last-value subtraction.
    """

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.enc_in
        self.target_index = self._infer_target_index(configs)

        patch_len = min(getattr(configs, "patch_len", 24), self.seq_len)
        patch_stride = max(1, getattr(configs, "patch_stride", 12))
        top_k_patches = max(1, getattr(configs, "top_k_patches", 8))
        hidden_size = getattr(configs, "hidden_size", 64)
        model_dim = getattr(configs, "model_dim", 64)
        lstm_layers = getattr(configs, "lstm_layers", 2)
        dropout = getattr(configs, "dropout", 0.1)
        fft_keep_ratio = getattr(configs, "fft_keep_ratio", 0.35)
        self.ablation = normalize_ablation(getattr(configs, "ablation", "none"))
        self.disable_frequency_denoising = (
            self.ablation == "no_frequency_denoising"
            or bool(getattr(configs, "disable_frequency_denoising", False))
        )
        self.disable_patch_selection = (
            self.ablation == "no_patch_selection"
            or bool(getattr(configs, "disable_patch_selection", False))
        )
        self.disable_gated_attention = (
            self.ablation == "no_gated_attention"
            or bool(getattr(configs, "disable_gated_attention", False))
        )

        self.revin = RevIN(self.input_dim)
        self.nlinear = nn.Linear(self.seq_len, self.pred_len)
        self.denoiser = (
            nn.Identity()
            if self.disable_frequency_denoising
            else FrequencyDenoiser(self.seq_len, self.input_dim, fft_keep_ratio)
        )
        if self.disable_patch_selection:
            self.patch_selector = PatchPassthrough(patch_len=patch_len, patch_stride=patch_stride)
        else:
            self.patch_selector = MLPPatchSelector(
                input_dim=self.input_dim,
                patch_len=patch_len,
                patch_stride=patch_stride,
                top_k=top_k_patches,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        self.patch_projection = nn.Linear(self.input_dim, model_dim)
        self.encoder = nn.LSTM(
            model_dim,
            hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.gated_attention = None if self.disable_gated_attention else GatedAttention(hidden_size, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.pred_len),
        )

    def _infer_target_index(self, configs) -> int:
        if hasattr(configs, "target_index"):
            return int(configs.target_index)
        if getattr(configs, "features", "MS") == "MS":
            return self.input_dim - 1
        return 0

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm, mean, std = self.revin.norm(x_enc)
        target_seq = x_norm[:, :, self.target_index]
        last = target_seq[:, -1:]
        baseline = self.nlinear(target_seq - last) + last

        x_denoised = self.denoiser(x_norm)
        selected, _patch_scores, _selected_indices = self.patch_selector(x_denoised)

        batch, top_k, patch_len, channels = selected.shape
        tokens = selected.reshape(batch, top_k * patch_len, channels)
        tokens = self.patch_projection(tokens)

        states, _ = self.encoder(tokens)
        if self.disable_gated_attention:
            context = states[:, -1, :]
        else:
            context, _attention, _gate = self.gated_attention(states)
        residual = self.head(context)
        pred = (baseline + residual).unsqueeze(-1)
        return self.revin.denorm(pred, mean, std, self.target_index)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        raise NotImplementedError("PatchGatedLSTM supports forecasting tasks only.")
