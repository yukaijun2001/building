import os
import csv


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

    # 准备要写入的数据行。metric_type 既可能是实验类型，也可能是普通 model_id。
    if metric_type == "input_length":
        m = seq_len
        metric_column = metric_type
    elif metric_type == "segment_length":
        m = pred_len
        metric_column = metric_type
    else:
        m = pred_len
        metric_column = "prediction_length"
    row = [dataset_name, m, mae_formatted, mse_formatted, mape_formatted, r2_formatted]

    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            writer.writerow(['dataset_name', metric_column, 'mae', 'mse', 'mape', 'r2'])
        writer.writerow(row)


def record_runtime_metrics(args, setting, training_profile=None, inference_profile=None):
    metrics_dir = os.path.abspath("metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    data_path = getattr(args, "data_path", "")
    dataset_name = data_path.split("_")[1] if "_" in data_path else data_path
    training_profile = training_profile or {}
    inference_profile = inference_profile or {}

    filepath = os.path.join(metrics_dir, "runtime_metrics.csv")
    file_exists = os.path.isfile(filepath)
    fieldnames = [
        "dataset_name",
        "model_id",
        "model",
        "setting",
        "seq_len",
        "label_len",
        "pred_len",
        "batch_size",
        "training_time_seconds",
        "inference_time_seconds",
        "parameter_size_mb",
        "total_parameters",
        "trainable_parameters",
        "training_gpu_memory_mb",
        "training_gpu_reserved_mb",
    ]

    def fmt(value):
        return "" if value is None else float(f"{value:.6f}")

    row = {
        "dataset_name": dataset_name,
        "model_id": getattr(args, "model_id", ""),
        "model": getattr(args, "model", ""),
        "setting": setting,
        "seq_len": getattr(args, "seq_len", ""),
        "label_len": getattr(args, "label_len", ""),
        "pred_len": getattr(args, "pred_len", ""),
        "batch_size": getattr(args, "batch_size", ""),
        "training_time_seconds": fmt(training_profile.get("training_time_seconds")),
        "inference_time_seconds": fmt(inference_profile.get("inference_time_seconds")),
        "parameter_size_mb": fmt(training_profile.get("parameter_size_mb") or inference_profile.get("parameter_size_mb")),
        "total_parameters": training_profile.get("total_parameters") or inference_profile.get("total_parameters") or "",
        "trainable_parameters": training_profile.get("trainable_parameters") or inference_profile.get("trainable_parameters") or "",
        "training_gpu_memory_mb": fmt(training_profile.get("training_gpu_memory_mb")),
        "training_gpu_reserved_mb": fmt(training_profile.get("training_gpu_reserved_mb")),
    }

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_result_data(preds, trues, data_path):
    import pandas as pd

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
