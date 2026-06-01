import argparse
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math

MODEL_ORDER = ["Ours", "TimeXer", "MaPL-LLM", "iTransformer", "TimeMixer", "PatchTST"]
OURS_MODEL = "PatchGatedLSTM"
MODEL_NAME_BY_DISPLAY = {
    "Ours": OURS_MODEL,
    "TimeXer": "TimeXer",
    "MaPL-LLM": "Times_llama",
    "iTransformer": "iTransformer",
    "TimeMixer": "TimeMixer",
    "PatchTST": "PatchTST",
}
DISPLAY_NAME_BY_MODEL = {model: display for display, model in MODEL_NAME_BY_DISPLAY.items()}


def ordered_model_names(experiments):
    model_names = [
        MODEL_NAME_BY_DISPLAY[display_name]
        for display_name in MODEL_ORDER
        if MODEL_NAME_BY_DISPLAY[display_name] in experiments
    ]

    missing = [
        MODEL_NAME_BY_DISPLAY[display_name]
        for display_name in MODEL_ORDER
        if MODEL_NAME_BY_DISPLAY[display_name] not in experiments
    ]
    if missing:
        print(f"Warning: Missing expected models: {missing}")

    return model_names


def display_model_name(model_name: str):
    return DISPLAY_NAME_BY_MODEL.get(model_name, model_name)


def is_times_llama_dir(dir_name: str):
    return "Times_Llama" in dir_name or "Times_llama" in dir_name


def parse_model_name(dir_name: str):
    if is_times_llama_dir(dir_name):
        return "Times_llama"

    parts = dir_name.split('_')
    for expected_model in MODEL_NAME_BY_DISPLAY.values():
        if expected_model in parts:
            return expected_model
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate scatter plots for energy consumption prediction.")
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/ykj/build/Time-Series-Library/results",
        help="Root directory of results",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/ykj/build/对比脚本/散点图",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--building",
        type=str,
        default="assembly",
        help="Building category (e.g., assembly, education, food, office, parking, public)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=150,
        help="Number of sampled points per model (0 means use all)",
    )
    parser.add_argument(
        "--axis_max",
        type=float,
        default=None,
        help="Max limit for x and y axes (e.g., 1000). If not set, auto-calculated.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scaling factor for data values (e.g., 100.0 to map 0-10 to 0-1000).",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=4.5,
        help="Shift value to add to data before scaling (to move negative values to positive).",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["scatter", "aggregate"],
        default="scatter",
        help="Type of plot to generate: scatter or aggregate performance plot.",
    )
    return parser.parse_args()

def find_experiments(results_root: Path, building: str):
    experiments = {}
    
    # Regex to match the building, seq_len, and pred_len
    # Pattern: ..._Hog_{building}_..._sl{seq_len}_..._pl{pred_len}_...
    # We also need to extract the model name.
    # Based on observation: long_term_forecast_Hog_{building}_{SubName}_{ModelName}_...
    
    if not results_root.exists():
        print(f"Error: Results root {results_root} does not exist.")
        return experiments

    print(f"Scanning {results_root} for building={building}...")

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        dir_name = exp_dir.name
        
        if f"Hog_{building}_" not in dir_name and not is_times_llama_dir(dir_name):
            continue

        # Try to extract model name from the directory name.
        # Current result dirs may put the model name before or after Hog_{building},
        # so match against the models we plan to plot instead of relying on one index.
        model_name = parse_model_name(dir_name)

        if model_name is None:
            continue

        true_path = exp_dir / "true.npy"
        pred_path = exp_dir / "pred.npy"

        if true_path.exists() and pred_path.exists():
            experiments[model_name] = {
                "true": true_path,
                "pred": pred_path,
                "path": exp_dir
            }
    
    return experiments

def plot_scatter(experiments, building, out_dir: Path, num_points: int, axis_max: float = None, scale: float = 1.0, shift: float = 0.0):
    if not experiments:
        print("No matching experiments found.")
        return

    model_names = ordered_model_names(experiments)
    if not model_names:
        print("No expected models found after ordering.")
        return

    num_models = len(model_names)

    
    
    cols = 3
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).ravel()
    
    # Global limits to keep scales consistent
    if axis_max is not None:
        limit = axis_max
    else:
        all_max = 0
        # First pass to find max value for limits
        for name in model_names:
            true_vals = (np.load(experiments[name]["true"]).flatten() + shift) * scale
            pred_vals = (np.load(experiments[name]["pred"]).flatten() + shift) * scale
            all_max = max(all_max, np.max(true_vals), np.max(pred_vals))
        limit = all_max * 1.1
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        true_path = experiments[name]["true"]
        pred_path = experiments[name]["pred"]
        
        y_true_raw = np.load(true_path).flatten()
        y_pred_raw = np.load(pred_path).flatten()
        
        # Apply shift and scale
        y_true = (y_true_raw + shift) * scale
        y_pred = (y_pred_raw + shift) * scale

        if num_points and num_points > 0 and len(y_true) > num_points:
            # Random sampling instead of linspace
            idx = np.random.choice(len(y_true), num_points, replace=False)
            y_true_plot = y_true[idx]
            y_pred_plot = y_pred[idx]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        
        ax.scatter(y_true_plot, y_pred_plot, color='mediumseagreen', s=10, alpha=0.7, label='Data')
        
        # Reference lines
        x_ref = np.linspace(0, limit, 100)
        
        # y = x (Ideal) - Blue Solid
        ax.plot(x_ref, x_ref, color='deepskyblue', linestyle='-', linewidth=2, label='Ideal')
        
        # y = 1.2x (+20%) - Blue Dashed
        ax.plot(x_ref, x_ref * 1.2, color='dodgerblue', linestyle='--', linewidth=1.5)
        
        # y = 0.8x (-20%) - Blue Dashed
        ax.plot(x_ref, x_ref * 0.8, color='dodgerblue', linestyle='--', linewidth=1.5)
        
        # Fill between +/- 20%
        ax.fill_between(x_ref, x_ref * 0.8, x_ref * 1.2, color='lightgray', alpha=0.3, label='±20% Error')
        
        # Formatting
        title = display_model_name(name)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Energy data (Actual)", fontsize=10)
        ax.set_ylabel("Energy data (Predicted)", fontsize=10)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.grid(True, alpha=0.3)
        
        # Tick formatting
        ax.tick_params(axis='both', which='major', labelsize=9)

    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"scatter_{building}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {out_file}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(diff) / (np.abs(y_true) + 1e-8)) * 100.0)
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true)))**2))
    if ss_tot <= 1e-12:
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {"MSE": mse, "MAE": mae, "MAPE": mape, "R2": r2}

def plot_aggregate(experiments, building, out_dir: Path, scale: float = 1.0, shift: float = 0.0):
    if not experiments:
        print("No matching experiments found.")
        return

    model_names = ordered_model_names(experiments)
    if not model_names:
        print("No expected models found after ordering.")
        return

    metrics_per_model = {}
    for name in model_names:
        true_path = experiments[name]["true"]
        pred_path = experiments[name]["pred"]
        y_true_raw = np.load(true_path)
        y_pred_raw = np.load(pred_path)
        metrics_per_model[name] = compute_metrics(y_true_raw, y_pred_raw)

    display_names = []
    for name in model_names:
        display_names.append(display_model_name(name))

    metric_keys = ["MSE", "MAE", "MAPE", "R2"]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]

    x = np.arange(len(model_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for idx, metric in enumerate(metric_keys):
        ax = axes[idx]
        values = [metrics_per_model[name][metric] for name in model_names]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"aggregate_{building}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved aggregate performance plot to {out_file}")

def main():
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    
    experiments = find_experiments(results_root, args.building)
    
    if not experiments:
        print("No experiments found matching criteria.")
        return

    print(f"Found {len(experiments)} models: {list(experiments.keys())}")
    if args.plot_type == "scatter":
        plot_scatter(experiments, args.building, out_dir, args.num_points, args.axis_max, args.scale, args.shift)
    else:
        plot_aggregate(experiments, args.building, out_dir, args.scale, args.shift)

if __name__ == "__main__":
    main()
