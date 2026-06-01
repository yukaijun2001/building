#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure training time, inference time, and parameter size without editing the framework."
    )

    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, required=True)

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="MS")
    parser.add_argument("--target", type=str, default="electricity")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--label_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--tv_dt", type=int, default=0)
    parser.add_argument("--tv_B", type=int, default=0)
    parser.add_argument("--tv_C", type=int, default=0)
    parser.add_argument("--use_D", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--dec_in", type=int, required=True)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--out_channels", type=int, default=None)
    parser.add_argument("--time_feature_dim", type=int, default=4)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--channel_independence", type=int, default=0)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=1)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="avg")
    parser.add_argument("--seg_len", type=int, default=96)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--des", type=str, default="Param")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--use_dtw", action="store_true", default=False)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", action="store_true", default=False)
    parser.add_argument("--scaling", action="store_true", default=False)
    parser.add_argument("--permutation", action="store_true", default=False)
    parser.add_argument("--randompermutation", action="store_true", default=False)
    parser.add_argument("--magwarp", action="store_true", default=False)
    parser.add_argument("--timewarp", action="store_true", default=False)
    parser.add_argument("--windowslice", action="store_true", default=False)
    parser.add_argument("--windowwarp", action="store_true", default=False)
    parser.add_argument("--rotation", action="store_true", default=False)
    parser.add_argument("--spawner", action="store_true", default=False)
    parser.add_argument("--dtwwarp", action="store_true", default=False)
    parser.add_argument("--shapedtwwarp", action="store_true", default=False)
    parser.add_argument("--wdba", action="store_true", default=False)
    parser.add_argument("--discdtw", action="store_true", default=False)
    parser.add_argument("--discsdtw", action="store_true", default=False)
    parser.add_argument("--extra_tag", type=str, default="")

    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--patch_stride", type=int, default=12)
    parser.add_argument("--top_k_patches", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--fft_keep_ratio", type=float, default=0.35)

    parser.add_argument("--node_dim", type=int, default=10)
    parser.add_argument("--gcn_depth", type=int, default=2)
    parser.add_argument("--gcn_dropout", type=float, default=0.3)
    parser.add_argument("--propalpha", type=float, default=0.3)
    parser.add_argument("--conv_channel", type=int, default=32)
    parser.add_argument("--skip_channel", type=int, default=32)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--pos", type=int, choices=[0, 1], default=1)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_device(args: argparse.Namespace) -> None:
    if args.use_gpu and args.gpu_type == "cuda" and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using GPU")
    elif args.use_gpu and args.gpu_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        args.device = torch.device("mps")
        print("Using MPS")
    else:
        args.use_gpu = False
        args.device = torch.device("cpu")
        print("Using CPU")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]


def synchronize(args: argparse.Namespace) -> None:
    if args.use_gpu and args.gpu_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif args.use_gpu and args.gpu_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def reset_gpu_memory_stats(args: argparse.Namespace) -> None:
    if args.use_gpu and args.gpu_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.empty_cache()


def training_gpu_memory_mb(args: argparse.Namespace) -> float:
    if args.use_gpu and args.gpu_type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_reserved(args.device) / (1024**2))
    return 0.0


def setting_name(args: argparse.Namespace, ii: int) -> str:
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
    )


def parameter_stats(model: torch.nn.Module) -> dict[str, float | int]:
    return {"parameter_size": int(sum(p.numel() for p in model.parameters()))}


def run_forward_pass(exp: Any, batch: tuple[Any, ...]) -> None:
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = torch.nan_to_num(batch_x.float().to(exp.device), nan=0.0, posinf=0.0, neginf=0.0)
    batch_y = torch.nan_to_num(batch_y.float().to(exp.device), nan=0.0, posinf=0.0, neginf=0.0)
    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)
    dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
    f_dim = -1 if exp.args.features == "MS" else 0
    _ = outputs[:, -exp.args.pred_len :, f_dim:].detach()


def measure_inference(exp: Any, args: argparse.Namespace) -> dict[str, float | int]:
    _test_data, test_loader = exp._get_data(flag="test")
    exp.model.eval()

    synchronize(args)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            run_forward_pass(exp, batch)
    synchronize(args)
    elapsed = time.perf_counter() - started

    return {"inference_time": float(elapsed)}


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "training_time",
        "inference_time",
        "parameter_size",
        "training_gpu_memory_mb",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    sys.path.insert(0, str(project_dir))
    os.chdir(project_dir)

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    set_seed(args.seed)
    configure_device(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = Exp_Long_Term_Forecast(args)
    stats = parameter_stats(exp.model)

    setting = setting_name(args, 0)
    training_time = 0.0
    training_memory = 0.0
    if args.is_training:
        print(f">>>>>>>start timed training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        reset_gpu_memory_stats(args)
        synchronize(args)
        started = time.perf_counter()
        exp.train(setting)
        synchronize(args)
        training_time = time.perf_counter() - started
        training_memory = training_gpu_memory_mb(args)
    else:
        checkpoint = Path(args.checkpoints) / setting / "checkpoint.pth"
        print(f"loading model from {checkpoint}")
        exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))

    print(f">>>>>>>start timed inference : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    inference_stats = measure_inference(exp, args)

    result = {
        "training_time": float(training_time),
        **inference_stats,
        **stats,
        "training_gpu_memory_mb": float(training_memory),
    }
    save_json(result, output_dir / "metrics.json")
    append_summary(Path(args.summary_csv), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
