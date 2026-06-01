import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

"""
python /home/ykj/能耗预测/energy-consumption-prediction/对比脚本/特征选择热力图.py  --annotate    --sort_by_importance
"""
def load_mask_weights(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    mask = np.asarray(obj.get("mask", []), dtype=np.float32)
    weights = np.asarray(obj.get("weights", []), dtype=np.float32)
    names = obj.get("feature_names", None)
    if names is None:
        names = [f"feat_{i}" for i in range(len(mask))]
    return mask, weights, names


def build_importance(mask, weights, names):
    mask = np.asarray(mask, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    d = min(mask.shape[0], weights.shape[0], len(names))
    mask = mask[:d]
    weights = weights[:d]
    names = list(names)[:d]

    importance = mask * weights

    # 按重要性排序（从大到小）
    order = np.argsort(-importance)
    importance_sorted = importance[order]
    names_sorted = [names[i] for i in order]
    return importance_sorted, names_sorted


def compute_corr_from_csv(csv_path, feature_names, sort_names=None):
    """
    从原始数据计算相关系数矩阵（Pearson），对角线=1（在方差不为0时）
    csv要求包含 feature_names 这些列
    """
    if pd is None:
        raise RuntimeError("未安装 pandas，请先 pip install pandas")

    df = pd.read_csv(csv_path)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少这些特征列: {missing}")

    df_feat = df[feature_names].astype(np.float32)
    df_feat = df_feat.dropna(how="any")
    if df_feat.shape[0] < 2:
        raise ValueError("有效样本数不足，无法计算相关系数矩阵，请检查缺失值或常数列")

    # 如果你希望按重要性排序后的顺序画相关性矩阵
    if sort_names is not None:
        idx = [feature_names.index(n) for n in sort_names]
        df_feat = df_feat.iloc[:, idx]
        names = list(sort_names)
    else:
        names = list(feature_names)

    # 相关系数矩阵
    X = df_feat.to_numpy(dtype=np.float32)
    corr = np.corrcoef(X, rowvar=False)

    # 数值稳定：如果某列方差为0，会出现 NaN；这里把 NaN 置 0，并强制对角线为 1
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    return corr, names


def compute_importance_outer(importance, names):
    """
    重要性外积矩阵：不是相关性矩阵，对角线不是 1（除非你强行改）
    """
    mat = np.outer(importance, importance)
    return mat, list(names)


def plot_heatmap_grid(mat, names, out_png, title, is_corr=False, annotate=True):
    n = len(names)
    fig_size = max(6, n * 0.6)

    plt.figure(figsize=(fig_size, fig_size))

    # 相关性矩阵建议固定范围 [-1, 1]，颜色更像你参考图
    if is_corr:
        im = plt.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    else:
        im = plt.imshow(mat, cmap="RdBu_r")

    # 画格子边框（更像“田字格”）
    plt.gca().set_xticks(np.arange(-0.5, n, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, n, 1), minor=True)
    plt.grid(which="minor", linestyle="-", linewidth=1)
    plt.gca().tick_params(which="minor", bottom=False, left=False)

    if annotate:
        # 特征很多时建议 annotate=False
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.xticks(ticks=np.arange(n), labels=names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(n), labels=names)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def resolve_mask_path(base_dir):
    candidates = [
        os.path.join(base_dir, "base_line", "Time-Series-Library", "mask_weights.json"),
        os.path.join(base_dir, "src", "outputs", "mask_weights.json"),
        os.path.join(base_dir, "outputs", "mask_weights.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"未找到 mask_weights.json，尝试路径: {candidates}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["corr", "importance"], default="corr",
                        help="corr=相关性矩阵(对角线=1)，importance=重要性外积矩阵(对角线≠1)")
    parser.add_argument("--csv_path", type=str, default="/home/ykj/能耗预测/energy-consumption-prediction/data/Hog_assembly_Colette.csv",
                        help="原始特征数据CSV路径（mode=corr 必填）")
    parser.add_argument("--out_png", type=str, default="/home/ykj/能耗预测/energy-consumption-prediction/对比脚本/热力图/corrdqn_feature_importance_heatmap.png", help="输出图片路径")
    parser.add_argument("--annotate", action="store_true", help="是否在格子里写数值（特征多时不建议）")
    parser.add_argument("--sort_by_importance", action="store_true",
                        help="mode=corr 时，是否按 mask*weight 重要性排序后再画相关性矩阵")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mask_path = resolve_mask_path(base_dir)

    mask, weights, names = load_mask_weights(mask_path)
    importance, names_sorted = build_importance(mask, weights, names)

    if args.mode == "corr":
        if not args.csv_path:
            raise ValueError("mode=corr 必须提供 --csv_path（原始特征数据CSV）")

        if args.sort_by_importance:
            corr, plot_names = compute_corr_from_csv(args.csv_path, list(names), sort_names=names_sorted)
        else:
            corr, plot_names = compute_corr_from_csv(args.csv_path, list(names), sort_names=None)

        plot_heatmap_grid(
            corr,
            plot_names,
            args.out_png,
            title="Pearson Correlation Heatmap ",
            is_corr=True,
            annotate=args.annotate,
        )
        print(f"保存相关性热力图到: {args.out_png}")

    else:  # importance
        mat, plot_names = compute_importance_outer(importance, names_sorted)
        plot_heatmap_grid(
            mat,
            plot_names,
            args.out_png,
            title="Feature Importance Heatmap (Outer Product, Not Correlation)",
            is_corr=False,
            annotate=args.annotate,
        )
        print(f"保存重要性外积热力图到: {args.out_png}")


if __name__ == "__main__":
    main()
