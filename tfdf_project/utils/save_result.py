import os
import csv

import pandas as pd


def record_metrics(data_path, metric_type, size, mae, mse, mape, r2):
    # 创建metrics目录（如果不存在）
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # 从data_path中提取数据集名称
    dataset_name = data_path.split("_")[1]

    # 格式化数值，保留6位小数
    seq_len = size[0]
    label_len = size[1]
    pred_len = size[2]
    mae_formatted = float(f"{mae:.6f}")
    mse_formatted = float(f"{mse:.6f}")
    mape_formatted = float(f"{mape:.6f}")
    r2_formatted = float(f"{r2:.6f}")

    # 构建CSV文件路径
    filename = f"{metric_type}.csv"
    filepath = os.path.join(metrics_dir, filename)

    # 准备要写入的数据行
    if metric_type == "input_length":
        m = seq_len
    elif metric_type == "segment_length":
        m = pred_len
    else:
        raise NotImplemented
    row = [dataset_name, m, mae_formatted, mse_formatted, mape_formatted, r2_formatted]

    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            writer.writerow(['dataset_name', f'{metric_type}', 'mae', 'mse', 'mape', 'r2'])
        writer.writerow(row)


def save_result_data(preds, trues, data_path):
    # 确保结果目录存在
    os.makedirs("result_data", exist_ok=True)
    dataset_name = data_path.split("_")[1]

    # 处理预测数据
    df_preds = pd.DataFrame(preds.squeeze())
    df_preds.columns = [i for i in range(df_preds.shape[1])]

    # 处理真实值数据
    df_trues = pd.DataFrame(trues.squeeze())
    df_trues.columns = [i for i in range(df_trues.shape[1])]

    # 保存到result_data目录
    preds_path = os.path.join("result_data", f"{dataset_name}_prediction.csv")
    trues_path = os.path.join("result_data", f"{dataset_name}_true.csv")

    df_preds.to_csv(preds_path, index=False)
    df_trues.to_csv(trues_path, index=False)

    print(f"预测结果已保存到: {preds_path}")
    print(f"真实值已保存到: {trues_path}")

    return preds_path, trues_path
