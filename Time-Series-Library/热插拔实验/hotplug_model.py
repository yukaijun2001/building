from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from layers.PatchGatedLSTM_layers import FrequencyDenoiser, GatedAttention, MLPPatchSelector, RevIN


class NLinearBaseline(nn.Module):
    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_seq = x[:, :, 0]
        last = target_seq[:, -1:]
        return self.linear(target_seq - last) + last


class TSLTargetBaseline(nn.Module):
    def __init__(self, configs, baseline_name: str) -> None:
        super().__init__()
        self.baseline_name = normalize_baseline_name(baseline_name)
        self.pred_len = configs.pred_len
        cfg = make_target_only_config(configs)

        if self.baseline_name == "patchtst":
            from models.PatchTST import Model as BaselineModel

            self.model = BaselineModel(
                cfg,
                patch_len=min(getattr(configs, "baseline_patch_len", 16), configs.seq_len),
                stride=getattr(configs, "baseline_patch_stride", 8),
            )
        elif self.baseline_name == "itransformer":
            from models.iTransformer import Model as BaselineModel

            self.model = BaselineModel(cfg)
        elif self.baseline_name == "timemixer":
            from models.TimeMixer import Model as BaselineModel

            if cfg.down_sampling_method is None:
                cfg.down_sampling_method = "avg"
            cfg.down_sampling_layers = max(1, int(getattr(cfg, "down_sampling_layers", 0)))
            cfg.down_sampling_window = max(2, int(getattr(cfg, "down_sampling_window", 1)))
            self.model = BaselineModel(cfg)
        elif self.baseline_name == "timexer":
            from models.TimeXer import Model as BaselineModel

            self.model = BaselineModel(cfg)
        else:
            raise ValueError(f"Unsupported TSL baseline: {baseline_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Several TSL baselines normalize their input with in-place ops. The
        # baseline still trains its own parameters; we just avoid backprop into
        # the shared normalized input tensor used by the residual branch.
        x = x.detach()
        out = self.model(x, None, None, None)
        if out.shape[-1] != 1:
            out = out[:, :, -1:]
        return out[:, -self.pred_len :, 0]


class MaPLLlamaTargetBaseline(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.pred_len = configs.pred_len
        jie_project = Path(getattr(configs, "mapl_project", "/home/ykj/build/jie_project")).resolve()
        if str(jie_project) not in sys.path:
            sys.path.insert(0, str(jie_project))

        from components.models.times_llama import TimesLlama

        cfg = make_mapl_config(configs)
        self.model = TimesLlama(cfg)

    def forward(self, x: torch.Tensor, x_mark_enc: torch.Tensor | None = None) -> torch.Tensor:
        if x_mark_enc is None:
            raise ValueError("MaPL-LLM baseline requires x_mark_enc from the building .pt file.")
        x = x.detach()
        out = self.model(x, x_mark_enc, None, None)
        if out.shape[-1] != 1:
            out = out[:, :, -1:]
        return out[:, -self.pred_len :, 0]


def normalize_baseline_name(value: str) -> str:
    key = str(value or "nlinear").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "nlinear": "nlinear",
        "mapl": "mapl_llm",
        "maplllm": "mapl_llm",
        "mapllama": "mapl_llm",
        "timesllama": "mapl_llm",
        "timesllm": "mapl_llm",
        "patchtst": "patchtst",
        "itransformer": "itransformer",
        "timemixer": "timemixer",
        "timexer": "timexer",
    }
    if key not in aliases:
        choices = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown hotplug baseline '{value}'. Choose one of: {choices}")
    return aliases[key]


def make_mapl_config(configs) -> SimpleNamespace:
    cfg = copy.copy(configs)
    cfg.seq_len = configs.seq_len
    cfg.label_len = configs.label_len
    cfg.pred_len = configs.pred_len
    cfg.gpu = getattr(configs, "gpu", 0)
    cfg.use_amp = bool(getattr(configs, "mapl_use_amp", getattr(configs, "use_amp", False)))
    cfg.llm_ckp_dir = getattr(configs, "llm_ckp_dir", "/home/ykj/build/llama_model")
    cfg.mlp_hidden_dim = getattr(configs, "mlp_hidden_dim", 1024)
    cfg.mlp_hidden_layers = getattr(configs, "mlp_hidden_layers", 2)
    cfg.mlp_activation = getattr(configs, "mlp_activation", "relu")
    cfg.num_experts = getattr(configs, "num_experts", 1)
    cfg.lambda_reg = getattr(configs, "lambda_reg", 0.01)
    return cfg


def make_target_only_config(configs) -> SimpleNamespace:
    cfg = copy.copy(configs)
    cfg.task_name = "long_term_forecast"
    cfg.enc_in = 1
    cfg.dec_in = 1
    cfg.c_out = 1
    cfg.features = "M"
    cfg.use_norm = getattr(configs, "baseline_use_norm", getattr(configs, "use_norm", 1))
    cfg.patch_len = min(getattr(configs, "baseline_patch_len", getattr(configs, "patch_len", 16)), configs.seq_len)
    cfg.channel_independence = getattr(configs, "channel_independence", 1)
    cfg.down_sampling_layers = getattr(configs, "down_sampling_layers", 0)
    cfg.down_sampling_window = getattr(configs, "down_sampling_window", 1)
    cfg.down_sampling_method = getattr(configs, "down_sampling_method", None)
    cfg.decomp_method = getattr(configs, "decomp_method", "moving_avg")
    cfg.moving_avg = getattr(configs, "moving_avg", 25)
    cfg.top_k = getattr(configs, "top_k", 5)
    cfg.embed = getattr(configs, "embed", "timeF")
    cfg.freq = getattr(configs, "freq", "h")
    cfg.dropout = getattr(configs, "dropout", 0.1)
    cfg.d_model = getattr(configs, "baseline_d_model", getattr(configs, "d_model", 128))
    cfg.n_heads = getattr(configs, "baseline_n_heads", getattr(configs, "n_heads", 4))
    cfg.e_layers = getattr(configs, "baseline_e_layers", getattr(configs, "e_layers", 2))
    cfg.d_ff = getattr(configs, "baseline_d_ff", getattr(configs, "d_ff", 256))
    cfg.factor = getattr(configs, "factor", 1)
    cfg.activation = getattr(configs, "activation", "gelu")
    return cfg


class HotPlugPatchGatedResidual(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.enc_in
        self.target_index = int(getattr(configs, "target_index", self.input_dim - 1))
        self.baseline_name = normalize_baseline_name(getattr(configs, "baseline", "nlinear"))
        self.disable_residual = bool(getattr(configs, "disable_residual", False))

        patch_len = min(getattr(configs, "patch_len", 24), self.seq_len)
        patch_stride = max(1, getattr(configs, "patch_stride", 12))
        top_k_patches = max(1, getattr(configs, "top_k_patches", 8))
        hidden_size = getattr(configs, "hidden_size", 64)
        model_dim = getattr(configs, "model_dim", 64)
        lstm_layers = getattr(configs, "lstm_layers", 2)
        dropout = getattr(configs, "dropout", 0.1)
        fft_keep_ratio = getattr(configs, "fft_keep_ratio", 0.35)

        self.revin = RevIN(self.input_dim)
        if self.baseline_name == "nlinear":
            self.baseline = NLinearBaseline(self.seq_len, self.pred_len)
        elif self.baseline_name == "mapl_llm":
            self.baseline = MaPLLlamaTargetBaseline(configs)
        else:
            self.baseline = TSLTargetBaseline(configs, self.baseline_name)

        if not self.disable_residual:
            self.denoiser = FrequencyDenoiser(self.seq_len, self.input_dim, fft_keep_ratio)
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
            self.gated_attention = GatedAttention(hidden_size, dropout)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.pred_len),
            )

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None) -> torch.Tensor:
        x_norm, mean, std = self.revin.norm(x_enc)
        target_only = x_norm[:, :, self.target_index : self.target_index + 1].clone()
        if self.baseline_name == "mapl_llm":
            baseline = self.baseline(target_only, x_mark_enc)
        else:
            baseline = self.baseline(target_only)
        if self.disable_residual:
            return self.revin.denorm(baseline.unsqueeze(-1), mean, std, self.target_index)

        x_denoised = self.denoiser(x_norm)
        selected, _patch_scores, _selected_indices = self.patch_selector(x_denoised)
        batch, top_k, patch_len, channels = selected.shape
        tokens = selected.reshape(batch, top_k * patch_len, channels)
        tokens = self.patch_projection(tokens)

        states, _ = self.encoder(tokens)
        context, _attention, _gate = self.gated_attention(states)
        residual = self.head(context)

        pred = (baseline + residual).unsqueeze(-1)
        return self.revin.denorm(pred, mean, std, self.target_index)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc)
        raise NotImplementedError("HotPlugPatchGatedResidual supports forecasting tasks only.")
