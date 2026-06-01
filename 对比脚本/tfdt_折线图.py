import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_NAME = "TFDFNet"
SEQ_LENS = [144, 168, 336, 504, 672, 720]
PRED_LEN = 24


def parse_exp_name(exp_name: str):
    building = None
    m_building = re.search(r"Hog_([^_]+)_", exp_name)
    if m_building:
        building = m_building.group(1)

    m_seq = re.search(r"sl(\d+)", exp_name)
    seq_len = int(m_seq.group(1)) if m_seq else None

    m_pred = re.search(r"pl(\d+)", exp_name)
    pred_len = int(m_pred.group(1)) if m_pred else None

    return building, seq_len, pred_len


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def load_metrics(exp_dir: Path):
    metrics_path = exp_dir / "metrics.npy"
    true_path = exp_dir / "true.npy"
    pred_path = exp_dir / "pred.npy"

    if metrics_path.exists():
        metrics = np.load(metrics_path)
        mae, mse, _rmse, mape, _mspe = metrics
    else:
        y_true = np.load(true_path)
        y_pred = np.load(pred_path)
        diff = y_pred.reshape(-1) - y_true.reshape(-1)
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff**2)
        mape = np.mean(np.abs(diff) / (np.abs(y_true.reshape(-1)) + 1e-8)) * 100.0

    y_true = np.load(true_path)
    y_pred = np.load(pred_path)
    r2 = compute_r2(y_true, y_pred)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def collect_tfdfnet_metrics(results_root: Path):
    stats = {}

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if MODEL_NAME not in exp_dir.name:
            continue

        building, seq_len, pred_len = parse_exp_name(exp_dir.name)
        if building is None or seq_len is None or pred_len is None:
            continue
        if pred_len != PRED_LEN or seq_len not in SEQ_LENS:
            continue

        true_path = exp_dir / "true.npy"
        pred_path = exp_dir / "pred.npy"
        if not (true_path.exists() and pred_path.exists()):
            continue

        try:
            metrics = load_metrics(exp_dir)
        except Exception as exc:
            print(f"跳过 {exp_dir.name}: {exc}")
            continue

        stats.setdefault(building, {})
        stats[building][seq_len] = metrics

    return stats


def add_timestamp(path: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def plot_curves(stats, out_path: Path):
    if not stats:
        raise RuntimeError("没有找到 TFDFNet pred_len=24 且 seq_len=144-720 的结果。")

    metric_keys = ["MAE", "MSE", "MAPE", "R2"]
    titles = ["MAE", "MSE", "MAPE", "R2"]
    ylabels = ["MAE", "MSE", "MAPE", "R2"]
    x_pos = {seq_len: idx for idx, seq_len in enumerate(SEQ_LENS)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for metric, title, ylabel, ax in zip(metric_keys, titles, ylabels, axes):
        for building in sorted(stats.keys()):
            seq_metrics = stats[building]
            x_vals = []
            y_vals = []
            for seq_len in SEQ_LENS:
                if seq_len not in seq_metrics:
                    continue
                x_vals.append(x_pos[seq_len])
                y_vals.append(seq_metrics[seq_len][metric])

            if x_vals:
                ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=building)

        ax.set_title(f"{MODEL_NAME} {title} Curves")
        ax.set_xlabel("Input length")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(len(SEQ_LENS))))
        ax.set_xticklabels([str(v) for v in SEQ_LENS])
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.86, 0.95),
            fontsize=9,
            frameon=True,
        )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot TFDFNet metrics over different seq_len values.")
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/ykj/build/Time-Series-Library/results",
        help="Time-Series-Library results directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/ykj/build/对比脚本/tfdf折线图",
        help="Output directory for the line chart",
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Do not add timestamp to output filename",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"结果目录不存在: {results_root}")

    stats = collect_tfdfnet_metrics(results_root)
    out_path = out_dir / f"{MODEL_NAME}_seq144-720_pred{PRED_LEN}.png"
    if not args.no_timestamp:
        out_path = add_timestamp(out_path)

    plot_curves(stats, out_path)
    print(f"已保存折线图到: {out_path}")
    print(f"已收集建筑: {sorted(stats.keys())}")


if __name__ == "__main__":
    main()
