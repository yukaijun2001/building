from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    model: str = "PatchGatedLSTM"
    data_dir: Path = Path("data")
    output_dir: Path = Path("code/outputs")
    target_col: str = "electricity"
    seq_len: int = 168
    pred_len: int = 24
    stride: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seg_len: int = 24
    patch_len: int = 24
    patch_stride: int = 12
    top_k_patches: int = 8
    hidden_size: int = 64
    model_dim: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1
    fft_keep_ratio: float = 0.35
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    limit_windows: int | None = None
    max_files: int | None = None
