"""
能耗预测模型对比图绘制脚本
可以选择画 train 或 test 集，支持多个建筑的批量绘制

常用命令：
1. 画默认模型在 test 集第 0 到 96 个点的预测曲线：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py

2. 画指定时间段，例如第 24 到 72 个点：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --start 24 --end 72

3. 只画某个模型：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --models PatchGatedLSTM --start 24 --end 72

4. 画多个模型，用英文逗号分隔模型名：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --models PatchGatedLSTM,TimeXer,PatchTST --start 24 --end 72

5. 只输出单模型图，不输出合并图：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --plot_mode single --start 24 --end 72

6. 只输出合并图，不输出单模型图：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --plot_mode combined --start 0 --end 96

7. 画 train 集曲线，需要结果目录里有 train_true.npy 和 train_pred.npy：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --split train --start 0 --end 96

8. 指定其它结果目录：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --results_dir /home/ykj/build/Time-Series-Library/results_拟合图 --start 0 --end 96

9. 让曲线显示得更密集：增大时间范围，或缩小图宽，例如：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --start 0 --end 336 --fig_width 8 --tick_step 24

10. 画完整 test 时间段：
   conda run -n time_series python /home/ykj/build/对比脚本/model_comparison_plot.py --start 0 --end all --fig_width 16 --tick_step 168

参数说明：
- --start：曲线起始点，下标从 0 开始，包含该点
- --end：曲线结束点，不包含该点；也可设为 all，表示画到预测结果最后一个点
- --models：模型名，默认 all，表示使用脚本 default_models 里的模型；多个模型用英文逗号分隔
- --split：数据集类型，可选 test 或 train，默认 test
- --plot_mode：输出模式，可选 single、combined、both，默认 both
- --fig_width：单个子图宽度，默认 12；数值越小，同样点数看起来越密集
- --fig_height：单个子图高度，默认 6
- --tick_step：x 轴刻度间隔，默认 6；时间范围较长时可设为 12、24、48
- --results_dir：预测结果根目录，默认 /home/ykj/build/Time-Series-Library/results
- --output_root：图片输出根目录，默认 /home/ykj/build/对比脚本/output
- --no_metrics：不在图中显示 MSE、RMSE、MAE、R²

输出位置：
- 单模型图：output_root/建筑名/模型名_split_tstart-end.png
- 合并图：output_root/建筑名/all_models_split_tstart-end.png
"""

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
    def __init__(self, results_dir, show_metrics=True, models=None, fig_width=12, fig_height=6, tick_step=6):
        self.results_dir = Path(results_dir)
        self.show_metrics = bool(show_metrics)
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.tick_step = tick_step

            # 'TimeXer',
            # 'Times_llama',
            # 'iTransformer',
            # 'TimeMixer',
            # 'PatchTST',

        default_models = [
            'PatchGatedLSTM',
            'TimeXer',
            'Times_llama',
            'iTransformer',
            'TimeMixer',
            'PatchTST',
        ]
        self.models = models if models else default_models
            # 'Hog_assembly_Colette',
            # 'Hog_education_Casandra',
            # 'Hog_food_Morgan',
            # 'Hog_office_Almeda',
            # 'Hog_parking_Jean',
            # 'Hog_public_Crystal',
        self.buildings = [

            'Hog_assembly_Colette'
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
        ticks = list(np.arange(start, end + 1, self.tick_step))
        if not ticks or ticks[-1] != end:
            ticks.append(end)
        ax.set_xlim(start, end)
        ax.set_xticks(ticks)

    def normalize_series(self, values):
        if values.ndim == 3:
            return values[:, 0, -1]
        if values.ndim == 2:
            return values[:, 0]
        return values

    def validate_time_range(self, start, end, series_length):
        if start < 0:
            raise ValueError("start 不能小于 0")
        if start >= series_length:
            raise ValueError(f"start={start} 超出序列长度 {series_length}")
        if end == "all":
            return series_length
        if end <= start:
            raise ValueError("end 必须大于 start")
        return min(end, series_length)

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

        true_values = self.normalize_series(true_values)
        pred_values = self.normalize_series(pred_values)
        real_end = self.validate_time_range(start, end, min(len(true_values), len(pred_values)))

        true_slice = true_values[start:real_end]
        pred_slice = pred_values[start:real_end]
        time_axis = np.arange(start, real_end)

        plt.figure(figsize=(self.fig_width, self.fig_height))
        plt.plot(time_axis, true_slice, label='GroundTruth', linewidth=2)
        plt.plot(time_axis, pred_slice, label='Prediction', linestyle='--', linewidth=2)

        title = self.get_display_name(model_name)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        self.format_x_axis(plt.gca(), start, real_end)

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
        cols = 2 if len(self.models) > 1 else 1
        rows = int(np.ceil(len(self.models) / cols))
        fig = plt.figure(figsize=(self.fig_width * cols, self.fig_height * rows))

        for i, model in enumerate(self.models):
            true, pred = self.load_prediction_data(model, building_name, split)
            if true is None:
                continue

            true = self.normalize_series(true)
            pred = self.normalize_series(pred)
            real_end = self.validate_time_range(start, end, min(len(true), len(pred)))
            t = np.arange(start, real_end)

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.plot(t, true[start:real_end], label='GroundTruth')
            ax.plot(t, pred[start:real_end], label='Prediction', linestyle='--')

            title = self.get_display_name(model)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            self.format_x_axis(ax, start, real_end)

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
    parser.add_argument("--models", type=str, default="all", help='逗号分隔的模型名，例如 "PatchGatedLSTM,TimeXer"；默认 all 表示使用脚本内全部模型')
    parser.add_argument("--split", type=str, choices=["test", "train"], default="test")
    parser.add_argument("--plot_mode", type=str, choices=["single", "combined", "both"], default="both")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=str, default="96")
    parser.add_argument("--fig_width", type=float, default=12)
    parser.add_argument("--fig_height", type=float, default=6)
    parser.add_argument("--tick_step", type=int, default=6)
    parser.add_argument("--no_metrics", action="store_true")
    args = parser.parse_args()

    selected_models = None
    if args.models.strip().lower() != "all":
        selected_models = [x.strip() for x in args.models.split(",") if x.strip()]

    plotter = ModelComparisonPlotter(
        args.results_dir,
        show_metrics=not args.no_metrics,
        models=selected_models,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        tick_step=args.tick_step,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.building_indices.strip().lower() == "all":
        building_indices = list(range(len(plotter.buildings)))
    else:
        building_indices = [int(x.strip()) for x in args.building_indices.split(",") if x.strip()]
    start = args.start
    end = args.end.strip().lower()
    if end != "all":
        end = int(end)

    for idx in building_indices:
        out_dir = output_root / plotter.buildings[idx]
        out_dir.mkdir(exist_ok=True)

        if args.plot_mode in ["single", "both"]:
            plotter.plot_all_models_for_one_building(
                idx, out_dir, start, end, split=args.split
            )

        if args.plot_mode in ["combined", "both"]:
            plotter.plot_all_models_in_one_figure(
                idx,
                out_dir / f"all_models_{args.split}_t{start}-{end}.png",
                start, end, split=args.split
            )


if __name__ == "__main__":
    main()
