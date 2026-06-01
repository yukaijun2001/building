#!/usr/bin/env python3
"""
遍历 results 目录，打印每个实验的 metrics.npy 指标
"""

import os
import numpy as np
from pathlib import Path

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot

def main():
    # 根据实际路径修改这里
    results_root = Path("/home/ykj/build/Time-Series-Library/results")

    if not results_root.exists():
        print(f"结果目录不存在: {results_root}")
        return

    # 遍历 results 下的所有子目录
    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        metrics_path = exp_dir / "metrics.npy"
        if not metrics_path.exists():
            # 没有 metrics.npy 的目录跳过
            continue

        try:
            metrics = np.load(metrics_path)
            # 你在 test() 中保存的顺序是: [mae, mse, rmse, mape, mspe]
            mae, mse, rmse, mape, mspe = metrics

            # 计算 R2
            pred_path = exp_dir / "pred.npy"
            true_path = exp_dir / "true.npy"
            r2 = 0.0
            if pred_path.exists() and true_path.exists():
                y_pred = np.load(pred_path)
                y_true = np.load(true_path)
                r2 = compute_r2(y_true, y_pred)

            print(f"\n实验目录: {exp_dir.name}")
            print(f"  MSE : {mse:.6f}")
            print(f"  MAE : {mae:.6f}")
            print(f"  MAPE: {mape:.6f}")
            print(f"  R²  : {r2:.6f}")

        except Exception as e:
            print(f"\n读取 {metrics_path} 失败: {e}")

if __name__ == "__main__":
    main()
