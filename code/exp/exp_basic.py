from __future__ import annotations

import torch

from data_provider.data_factory import data_provider
from data_provider.data_loader import SplitData
from models.PatchGatedLSTM import (
    PatchGatedLSTMForecaster,
    PatchGatedSegRNNForecaster,
)
from utils.config import ExperimentConfig
from utils.tools import get_device


class ExpBasic:
    def __init__(self, cfg: ExperimentConfig, split: SplitData) -> None:
        self.cfg = cfg
        self.split = split
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> torch.nn.Module:
        patch_kwargs = {
            "seq_len": self.cfg.seq_len,
            "pred_len": self.cfg.pred_len,
            "input_dim": len(self.split.feature_cols),
            "target_index": self.split.target_index,
            "patch_len": self.cfg.patch_len,
            "patch_stride": self.cfg.patch_stride,
            "top_k_patches": self.cfg.top_k_patches,
            "hidden_size": self.cfg.hidden_size,
            "model_dim": self.cfg.model_dim,
            "lstm_layers": self.cfg.lstm_layers,
            "dropout": self.cfg.dropout,
            "fft_keep_ratio": self.cfg.fft_keep_ratio,
        }
        if self.cfg.model == "PatchGatedSegRNN":
            return PatchGatedSegRNNForecaster(**patch_kwargs, seg_len=self.cfg.seg_len)
        if self.cfg.model != "PatchGatedLSTM":
            raise ValueError(f"Unknown model: {self.cfg.model}")
        return PatchGatedLSTMForecaster(**patch_kwargs)

    def _acquire_device(self) -> torch.device:
        device = get_device(self.cfg.device)
        print(f"Use device: {device}")
        return device

    def _get_data(self, flag: str):
        return data_provider(self.split, flag, self.cfg)

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
