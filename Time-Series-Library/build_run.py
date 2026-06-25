from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from data_provider.building_energy import prepare_building_split
from exp.exp_building_energy import Exp_Building_Energy


METRIC_NAMES = ["mse", "mae", "mape", "r2"]


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
        raise ValueError(f"Unknown ablation '{value}'. Choose one of: {choices}")
    return ABLATION_ALIASES[key]


def parse_args():
    parser = argparse.ArgumentParser(description="BuildingEnergy runner for Time-Series-Library")

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="PatchGatedLSTM_168_24")
    parser.add_argument("--model", type=str, default="PatchGatedLSTM")
    parser.add_argument("--data", type=str, default="BuildingEnergy")
    parser.add_argument("--root_path", type=str, default="../data")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--features", type=str, default="MS")
    parser.add_argument("--target", type=str, default="electricity")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--source_output_dir", type=str, default="./outputs/source_PatchGatedLSTM_168_24")
    parser.add_argument("--inverse", action="store_true", default=False, help="save test predictions in original target scale")

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--label_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--patch_stride", type=int, default=12)
    parser.add_argument("--top_k_patches", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fft_keep_ratio", type=float, default=0.35)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        help="PatchGatedLSTM ablation: none, no_frequency_denoising, no_patch_selection, no_gated_attention.",
    )
    parser.add_argument("--disable_frequency_denoising", action="store_true")
    parser.add_argument("--disable_patch_selection", action="store_true")
    parser.add_argument("--disable_gated_attention", action="store_true")

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--distil", action="store_false", default=True)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--dec_in", type=int, default=1)
    parser.add_argument("--c_out", type=int, default=1)
    args = parser.parse_args()
    args.ablation = normalize_ablation(args.ablation)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_device(args) -> None:
    if torch.cuda.is_available() and args.use_gpu and args.gpu_type == "cuda":
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using GPU")
    else:
        args.device = torch.device("cpu")
        args.use_gpu = False
        print("Using CPU")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(id_) for id_ in args.devices.split(",")]
        args.gpu = args.device_ids[0]


def setting_name(args, building_name: str, ii: int) -> str:
    ablation_suffix = "" if args.ablation == "none" else f"_abl{args.ablation}"
    return (
        f"{args.task_name}_{args.model_id}_{building_name}_{args.model}_{args.data}_ft{args.features}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_plen{args.patch_len}_pstride{args.patch_stride}_topkp{args.top_k_patches}"
        f"_hs{args.hidden_size}_mdim{args.model_dim}_lstm{args.lstm_layers}"
        f"_fft{args.fft_keep_ratio}{ablation_suffix}_seed{args.seed}_{args.des}_{ii}"
    )


def summarize_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([r[metric] for r in results.values()]))
        for metric in METRIC_NAMES
        if results and all(metric in r for r in results.values())
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_device(args)

    root_path = Path(args.root_path)
    if args.data_path:
        csv_files = [root_path / args.data_path]
    else:
        csv_files = sorted(root_path.glob("*.csv"))
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {root_path}")

    source_output_dir = Path(args.source_output_dir)
    source_output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), source_output_dir / "config.json")

    results = {}
    for ii in range(args.itr):
        for path in csv_files:
            building_name = path.stem
            print(f"\n=== {building_name} ===")
            split = prepare_building_split(
                path=path,
                target_col=args.target,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                stride=args.stride,
                limit_windows=args.limit_windows,
            )
            args.enc_in = len(split.feature_cols)
            args.dec_in = len(split.feature_cols)
            args.c_out = 1
            args.target_index = split.target_index

            exp = Exp_Building_Energy(args, split)
            setting = setting_name(args, building_name, ii)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            if args.is_training:
                exp.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            result = exp.test(setting, test=0 if args.is_training else 1)
            exp.save_source_outputs(building_name, source_output_dir)

            public_result = {k: result[k] for k in METRIC_NAMES}
            results[building_name] = public_result
            save_json({**results, "_mean": summarize_results(results)}, source_output_dir / "metrics.json")
            print(f"{building_name} test: {public_result}")

    results["_mean"] = summarize_results(results)
    save_json(results, source_output_dir / "metrics.json")
    print("\n=== Mean metrics ===")
    print(results["_mean"])


if __name__ == "__main__":
    main()
