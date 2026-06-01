import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_building_and_lens(exp_name: str):
    """
    从实验目录名中解析建筑类别、输入长度(seq_len, slXXX)和预测长度(pred_len, plXXX)。
    例：
    long_term_forecast_Hog_assembly_Colette_..._sl168_ll48_pl24_... -> (assembly, 168, 24)
    """
    bld = None
    m_bld = re.search(r"Hog_([^_]+)_", exp_name)
    if m_bld:
        bld = m_bld.group(1)

    m_sl = re.search(r"sl(\d+)", exp_name)
    seq_len = int(m_sl.group(1)) if m_sl else None

    m_pl = re.search(r"pl(\d+)", exp_name)
    pred_len = int(m_pl.group(1)) if m_pl else None

    return bld, seq_len, pred_len


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def smooth_series(values, method: str, window: int, alpha: float):
    arr = np.asarray(values, dtype=np.float64)
    if method == "none" or arr.size <= 1:
        return values
    if method == "ma":
        w = int(window)
        if w <= 1:
            return values
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        padded = np.pad(arr, (pad_left, pad_right), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / float(w)
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.tolist()
    if method == "ema":
        a = float(alpha)
        if not (0.0 < a <= 1.0):
            return values
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, arr.size):
            out[i] = a * arr[i] + (1.0 - a) * out[i - 1]
        return out.tolist()
    return values


def add_timestamp_suffix(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def collect_metrics(results_root: Path, axis: str):
    """
    遍历 results 目录，收集每个建筑在不同横轴取值下的 MAE/MSE/MAPE/R^2。
    axis='seq' 时：横轴为输入长度 seq_len（slXXX）；
    axis='pred' 时：横轴为预测长度 pred_len（plXXX）。
    返回: dict[building][x_value] = {"MAE":..., "MSE":..., "MAPE":..., "R2":...}
    """
    stats: dict[str, dict[int, dict[str, float]]] = {}

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        bld, seq_len, pred_len = parse_building_and_lens(exp_dir.name)
        if bld is None:
            continue

        if axis == "seq":
            x_val = seq_len
        else:
            x_val = pred_len

        if x_val is None:
            continue

        metrics_path = exp_dir / "metrics.npy"
        pred_path = exp_dir / "pred.npy"
        true_path = exp_dir / "true.npy"
        if not (metrics_path.exists() and pred_path.exists() and true_path.exists()):
            continue

        try:
            metrics = np.load(metrics_path)
            mae, mse, rmse, mape, mspe = metrics
            y_pred = np.load(pred_path)
            y_true = np.load(true_path)
            r2 = compute_r2(y_true, y_pred)
        except Exception:
            continue

        stats.setdefault(bld, {})
        stats[bld][x_val] = {
            "MAE": float(mae),
            "MSE": float(mse),
            "MAPE": float(mape),
            "R2": float(r2),
        }

    return stats


def plot_metrics_curves(
    stats: dict,
    out_path: Path,
    axis_label: str,
    axis: str,
    smooth: str,
    smooth_window: int,
    smooth_alpha: float,
    smooth_buildings: set,
):
    """
    绘制四个指标随给定横轴变化的折线图（所有建筑共用一张 2x2 图）。
    """
    if not stats:
        raise RuntimeError("未在 results 目录中解析到任何指标，请检查路径或实验结果。")

    metric_keys = ["MAE", "MSE", "MAPE", "R2"]
    titles = ["MAE Curves", "MSE Curves", "MAPE Curves", "R\u00b2 Curves"]
    ylabels = ["MAE", "MSE", "MAPE", "R\u00b2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    buildings = sorted(stats.keys())
    markers = ["o"]

    seq_ticks = [24, 48, 72, 96, 120, 144, 168, 336, 504, 672, 720] if axis == "seq" else None
    pred_ticks = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168] if axis == "pred" else None

    seq_pos = {v: i for i, v in enumerate(seq_ticks)} if seq_ticks is not None else None
    pred_pos = {v: i for i, v in enumerate(pred_ticks)} if pred_ticks is not None else None

    for idx_metric, (metric, title, ylabel) in enumerate(zip(metric_keys, titles, ylabels)):
        ax = axes[idx_metric]
        for idx_bld, bld in enumerate(buildings):
            seq_dict = stats[bld]
            raw_x_vals = sorted(seq_dict.keys())

            if axis == "seq" and seq_pos is not None:
                x_vals = []
                values = []
                for x in raw_x_vals:
                    if x not in seq_pos:
                        continue
                    x_vals.append(seq_pos[x])
                    values.append(seq_dict[x][metric])
            elif axis == "pred" and pred_pos is not None:
                x_vals = []
                values = []
                for x in raw_x_vals:
                    if x not in pred_pos:
                        continue
                    x_vals.append(pred_pos[x])
                    values.append(seq_dict[x][metric])
            else:
                x_vals = raw_x_vals
                values = [seq_dict[x][metric] for x in x_vals]

            if metric in {"MAPE", "R2"} and smooth != "none":
                if ("all" in smooth_buildings) or (bld in smooth_buildings):
                    values = smooth_series(values, method=smooth, window=smooth_window, alpha=smooth_alpha)

            ax.plot(x_vals, values, marker=markers[idx_bld % len(markers)], label=bld)

        ax.set_title(title)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if axis == "seq" and seq_ticks is not None:
            ax.set_xticks(list(range(len(seq_ticks))))
            ax.set_xticklabels([str(v) for v in seq_ticks])
        if axis == "pred" and pred_ticks is not None:
            ax.set_xticks(list(range(len(pred_ticks))))
            ax.set_xticklabels([str(v) for v in pred_ticks])

    all_handles = []
    all_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_labels:
                all_handles.append(h)
                all_labels.append(l)
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(0.86, 0.95),
            fontsize=8,
            frameon=True,
        )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/ykj/build/Time-Series-Library/results",
        help="实验结果根目录（包含各个 setting 子目录）",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="/home/ykj/build/对比脚本/折线图/metrics_curves.png",
        help="输出折线图路径",
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["seq", "pred"],
        default="pred",
        help="横轴类型：seq=输入长度(seq_len)，pred=预测长度(pred_len)",
    )
    parser.add_argument(
        "--smooth",
        type=str,
        choices=["none", "ma", "ema"],
        default="ma",
        help="是否对 R² 曲线做平滑：none=不平滑，ma=滑动平均，ema=指数滑动平均",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="平滑窗口（仅 ma 有效）",
    )
    parser.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.99,
        help="平滑系数（仅 ema 有效，(0,1]）",
    )
    parser.add_argument(
        "--smooth_buildings",
        type=str,
        default="education",
        help="哪些建筑的 R² 需要平滑：education 或 education,assembly 或 all",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"结果目录不存在: {results_root}")

    stats = collect_metrics(results_root, axis=args.x_axis)
    axis_label = "Input length" if args.x_axis == "seq" else "Prediction length"
    smooth_buildings = {s.strip() for s in str(args.smooth_buildings).split(",") if s.strip()}
    out_png = add_timestamp_suffix(Path(args.out_png))
    plot_metrics_curves(
        stats,
        out_png,
        axis_label=axis_label,
        axis=args.x_axis,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
        smooth_alpha=args.smooth_alpha,
        smooth_buildings=smooth_buildings,
    )
    print(f"已保存折线图到: {out_png}")


if __name__ == "__main__":
    main()
