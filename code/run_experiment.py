from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_provider.data_factory import prepare_split
from exp.exp_main import fit_building
from utils.config import ExperimentConfig
from utils.tools import ensure_dir, save_json, set_seed


METRIC_NAMES = [
    "mse",
    "mae",
    "mape",
    "r2",
]


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Building energy forecasting experiment")
    parser.add_argument(
        "--model",
        choices=["PatchGatedLSTM", "PatchGatedSegRNN"],
        default="PatchGatedLSTM",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("code/outputs"))
    parser.add_argument("--target-col", default="electricity")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seg-len", type=int, default=24)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=12)
    parser.add_argument("--top-k-patches", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fft-keep-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig(**vars(args))
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    positive_ints = {
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "stride": cfg.stride,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "seg_len": cfg.seg_len,
        "patch_len": cfg.patch_len,
        "patch_stride": cfg.patch_stride,
        "top_k_patches": cfg.top_k_patches,
        "hidden_size": cfg.hidden_size,
        "model_dim": cfg.model_dim,
        "lstm_layers": cfg.lstm_layers,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if cfg.patch_len > cfg.seq_len:
        raise ValueError("patch_len must be less than or equal to seq_len")
    if cfg.model == "PatchGatedSegRNN":
        if cfg.seq_len % cfg.seg_len != 0:
            raise ValueError(f"seq_len must be divisible by seg_len when using {cfg.model}")
        if cfg.pred_len % cfg.seg_len != 0:
            raise ValueError(f"pred_len must be divisible by seg_len when using {cfg.model}")
    if not 0 < cfg.fft_keep_ratio <= 1:
        raise ValueError("fft_keep_ratio must be in (0, 1]")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def summarize_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([r[metric] for r in results.values()]))
        for metric in METRIC_NAMES
        if results and all(metric in r for r in results.values())
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    save_json(asdict(cfg), cfg.output_dir / "config.json")

    csv_files = sorted(cfg.data_dir.glob("*.csv"))
    if cfg.max_files is not None:
        csv_files = csv_files[: cfg.max_files]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cfg.data_dir}")

    print(f"Found {len(csv_files)} building files")

    results = {}
    for path in csv_files:
        name = path.stem
        print(f"\n=== {name} ===")
        split = prepare_split(
            path=path,
            target_col=cfg.target_col,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            stride=cfg.stride,
            limit_windows=cfg.limit_windows,
        )
        result = fit_building(cfg, name, split, cfg.output_dir)
        results[name] = result
        save_json({**results, "_mean": summarize_results(results)}, cfg.output_dir / "metrics.json")
        print(f"{name} test: {result}")

    summary = summarize_results(results)
    results["_mean"] = summary
    save_json(results, cfg.output_dir / "metrics.json")
    print("\n=== Mean metrics ===")
    print(summary)


if __name__ == "__main__":
    main()
