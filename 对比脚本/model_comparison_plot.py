"""
能耗预测模型对比图绘制脚本
可以选择画 train 或 test 集，支持多个建筑的批量绘制
"""

import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelComparisonPlotter:
    def __init__(self, results_dir, show_metrics=True):
        self.results_dir = Path(results_dir)
        self.show_metrics = bool(show_metrics)

        self.models = [
            'PatchGatedLSTM',
            'TimeXer',
            'Times_llama',
            'iTransformer',
            'TimeMixer',
            'PatchTST',
        ]

        self.buildings = [
            'Hog_assembly_Colette',
            'Hog_education_Casandra',
            'Hog_food_Morgan',
            'Hog_office_Almeda',
            'Hog_parking_Jean',
            'Hog_public_Crystal',
        ]

    # ========= 指标 =========
    def compute_metrics(self, true, pred):
        mse = mean_squared_error(true, pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        return mse, rmse, mae, r2

    def get_display_name(self, model_name):
        aliases = {
            "PatchGatedLSTM": "Ours",
            "Times_llama": "MaPL-LLM",
        }
        return aliases.get(model_name, model_name)

    def load_prediction_data(self, model_name, building_name, split="test"):
        all_dirs = self.find_result_dirs(model_name, building_name)

        if split == "train":
            true_name = 'train_true.npy'
            pred_name = 'train_pred.npy'
        else:
            true_name = 'true.npy'
            pred_name = 'pred.npy'

        for result_dir in all_dirs:
            true_file = result_dir / true_name
            pred_file = result_dir / pred_name
            if true_file.exists() and pred_file.exists():
                true_values = np.load(true_file)
                pred_values = np.load(pred_file)
                return self.squeeze_prediction_arrays(true_values, pred_values)

        print(f"警告: 未找到 {model_name} 在 {building_name} 上的 {split} 预测结果")
        return None, None

    def find_result_dirs(self, model_name, building_name):
        patterns = [
            f"long_term_forecast_*_{building_name}_{model_name}_*",
            f"long_term_forecast_{building_name}_{model_name}*",
            f"*{building_name}*{model_name}*",
        ]
        if model_name == "Times_llama":
            patterns.extend([
                f"long_term_forecast_*_{building_name}_Times_Llama_*",
                f"*{building_name}*Times_Llama*",
                "long_term_forecast_Electricity_*_Times_Llama_building_single_*",
                "*Times_Llama*",
            ])

        seen = set()
        dirs = []
        for pattern in patterns:
            for result_dir in sorted(self.results_dir.glob(pattern)):
                if result_dir.is_dir() and result_dir not in seen:
                    seen.add(result_dir)
                    dirs.append(result_dir)
        return dirs

    def squeeze_prediction_arrays(self, true_values, pred_values):
        if true_values.ndim == 3:
            true_values = true_values.squeeze()
        if pred_values.ndim == 3:
            pred_values = pred_values.squeeze()
        return true_values, pred_values

    def format_x_axis(self, ax, start, end):
        ticks = list(np.arange(start, end + 1, 6))
        if not ticks or ticks[-1] != end:
            ticks.append(end)
        ax.set_xlim(start, end)
        ax.set_xticks(ticks)

    # ========= 单模型 =========
    def plot_single_model_comparison(
        self, model_name, building_index=0,
        save_path=None, start=0, end=96, split="test"
    ):
        building_name = self.buildings[building_index]
        true_values, pred_values = self.load_prediction_data(
            model_name, building_name, split
        )

        if true_values is None:
            return

        if true_values.ndim == 3:
            true_values = true_values[:, 0, -1]
        elif true_values.ndim == 2:
            true_values = true_values[:, 0]

        if pred_values.ndim == 3:
            pred_values = pred_values[:, 0, -1]
        elif pred_values.ndim == 2:
            pred_values = pred_values[:, 0]

        real_end = min(end, len(true_values), len(pred_values))

        true_slice = true_values[start:real_end]
        pred_slice = pred_values[start:real_end]
        time_axis = np.arange(start, real_end)

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, true_slice, label='GroundTruth', linewidth=2)
        plt.plot(time_axis, pred_slice, label='Prediction', linestyle='--', linewidth=2)

        title = self.get_display_name(model_name)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        self.format_x_axis(plt.gca(), start, end)

        # ✅ 固定右上角
        plt.legend(loc='upper right')

        if self.show_metrics:
            mse, rmse, mae, r2 = self.compute_metrics(true_slice, pred_slice)
            plt.text(
                0.02, 0.98,
                f"MSE : {mse:.4f}\nRMSE: {rmse:.4f}\nMAE : {mae:.4f}\nR²  : {r2:.4f}",
                transform=plt.gca().transAxes,
                ha='left', va='top',
                bbox=dict(boxstyle="round", fc="white", alpha=0.7)
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ========= 单建筑所有模型 =========
    def plot_all_models_for_one_building(
        self, building_index=0,
        save_dir=None, start=0, end=96, split="test"
    ):
        for model in self.models:
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"{model}_{split}_t{start}-{end}.png"

            self.plot_single_model_comparison(
                model, building_index, save_path, start, end, split
            )

    # ========= 合并图 =========
    def plot_all_models_in_one_figure(
        self, building_index=0,
        save_path=None, start=0, end=96, split="test"
    ):
        building_name = self.buildings[building_index]
        fig = plt.figure(figsize=(20, 15))

        for i, model in enumerate(self.models):
            true, pred = self.load_prediction_data(model, building_name, split)
            if true is None:
                continue

            if true.ndim == 3:
                true = true[:, 0, -1]
            elif true.ndim == 2:
                true = true[:, 0]

            if pred.ndim == 3:
                pred = pred[:, 0, -1]
            elif pred.ndim == 2:
                pred = pred[:, 0]

            real_end = min(end, len(true), len(pred))
            t = np.arange(start, real_end)

            ax = fig.add_subplot(3, 2, i + 1)
            ax.plot(t, true[start:real_end], label='GroundTruth')
            ax.plot(t, pred[start:real_end], label='Prediction', linestyle='--')

            title = self.get_display_name(model)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            self.format_x_axis(ax, start, end)

            # ✅ 固定右上角
            ax.legend(loc='upper right', fontsize=10)

            if self.show_metrics:
                mse, rmse, mae, r2 = self.compute_metrics(
                    true[start:real_end], pred[start:real_end]
                )
                ax.text(
                    0.02, 0.98,
                    f"MSE:{mse:.4f}\nRMSE:{rmse:.4f}\nMAE:{mae:.4f}\nR²:{r2:.4f}",
                    transform=ax.transAxes,
                    ha='left', va='top',
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7),
                    fontsize=9
                )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/ykj/build/Time-Series-Library/results")
    parser.add_argument("--output_root", type=str, default="/home/ykj/build/对比脚本/output")
    parser.add_argument(
        "--building_indices",
        type=str,
        default="0",
        help='逗号分隔的建筑索引，例如 "0,1,2"；默认 all 表示绘制全部建筑',
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=96)
    parser.add_argument("--no_metrics", action="store_true")
    args = parser.parse_args()

    plotter = ModelComparisonPlotter(
        args.results_dir,
        show_metrics=not args.no_metrics,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.building_indices.strip().lower() == "all":
        building_indices = list(range(len(plotter.buildings)))
    else:
        building_indices = [int(x.strip()) for x in args.building_indices.split(",") if x.strip()]
    start, end = args.start, args.end

    for idx in building_indices:
        out_dir = output_root / plotter.buildings[idx]
        out_dir.mkdir(exist_ok=True)

        plotter.plot_all_models_for_one_building(
            idx, out_dir, start, end, split="test"
        )

        plotter.plot_all_models_in_one_figure(
            idx,
            out_dir / f"all_models_test_t{start}-{end}.png",
            start, end, split="test"
        )


if __name__ == "__main__":
    main()
