#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/../../.." && pwd)
project_dir=$(cd "${tsl_dir}/.." && pwd)

model_name=PatchGatedLSTM
root_path_name=${ROOT_PATH:-${project_dir}/data}
target_col_name=${TARGET:-dayahead_price}
random_seed=${SEED:-42}
# 数据采样频率；15min 表示按 CSV 中的 15 分钟间隔建模。
freq_name=${FREQ:-15min}

# 输入历史点数。15min 数据下，672 个点表示 7 天。
seq_lens=${SEQ_LENS:-${SEQ_LEN:-"2688"}}
# 解码器标签长度，保留这个参数是为了兼容 TSL 参数；BuildingEnergy 流程构造窗口时不使用它。
label_len=${LABEL_LEN:-24}
# 每个 patch 的点数。15min 数据下，24 个点表示 6 小时。
patch_len=${PATCH_LEN:-24}
# 相邻 patch 的滑动步长，默认取 patch_len 的一半；也可用 PATCH_STRIDE 手动覆盖。
# patch_stride=${PATCH_STRIDE:-$((patch_len / 2))}
patch_stride=${PATCH_STRIDE:-12}
# 从所有候选 patch 中选择的高分 patch 数量。
top_k_patches=${TOP_K_PATCHES:-8}
# patch 选择器 MLP 和预测编码器 LSTM 使用的隐藏维度。
hidden_size=${HIDDEN_SIZE:-8}
# 预测编码器 LSTM 的堆叠层数。
lstm_layers=${LSTM_LAYERS:-1}
# patch token 进入预测编码器 LSTM 前的投影维度。
model_dim=${MODEL_DIM:-128}
# 频域去噪模块保留的低频 FFT 分量比例。
fft_keep_ratio=${FFT_KEEP_RATIO:-1}

# 训练、验证和测试 DataLoader 的批大小。
batch_size=${BATCH_SIZE:-64}
# DataLoader 的工作进程数，0 表示在主进程中加载数据。
num_workers=${NUM_WORKERS:-0}
# 最大训练轮数，实际训练可能会被早停提前结束。
train_epochs=${EPOCHS:-1000}
# 早停耐心值，表示验证集连续多少轮没有提升后停止训练。
patience=${PATIENCE:-3}
# AdamW 优化器的学习率。
learning_rate=${LR:-0.0001}
# AdamW 优化器的权重衰减，用于类似 L2 的正则化。
weight_decay=${WEIGHT_DECAY:-0.0001}
# 梯度裁剪的最大范数；设置为 0 或负数时关闭梯度裁剪。
grad_clip=${GRAD_CLIP:-1.0}
# 设备选择，设置 DEVICE=cpu 可强制使用 CPU；否则由 build_run.py 在可用时自动选择 GPU。
device=${DEVICE:-auto}

shopt -s nullglob
csv_files=("${root_path_name}"/*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
  echo "No CSV files found in ${root_path_name}" >&2
  echo "Set ROOT_PATH=/path/to/data if your dataset lives elsewhere." >&2
  exit 1
fi

cd "${tsl_dir}"

gpu_args=()
if [ "${device}" = "cpu" ]; then
  gpu_args+=(--no_use_gpu)
fi
# 12 24 36 48 60 72 84 96 108 120 132 144 156 168
for seq_len in ${seq_lens}
do
  for pred_len in ${PRED_LENS:-96 }
  do
    model_id_name=${model_name}_${seq_len}_${pred_len}
    "${PYTHON:-python}" -u build_run_15min.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${root_path_name}" \
      --source_output_dir "${tsl_dir}/outputs/source_${model_id_name}" \
      --model_id "${model_id_name}" \
      --model PatchGatedLSTM \
      --data BuildingEnergy \
      --features MS \
      --target "${target_col_name}" \
      --freq "${freq_name}" \
      --seq_len "${seq_len}" \
      --label_len "${label_len}" \
      --pred_len "${pred_len}" \
      --patch_len "${patch_len}" \
      --patch_stride "${patch_stride}" \
      --top_k_patches "${top_k_patches}" \
      --hidden_size "${hidden_size}" \
      --lstm_layers "${lstm_layers}" \
      --model_dim "${model_dim}" \
      --fft_keep_ratio "${fft_keep_ratio}" \
      --batch_size "${batch_size}" \
      --num_workers "${num_workers}" \
      --train_epochs "${train_epochs}" \
      --patience "${patience}" \
      --learning_rate "${learning_rate}" \
      --weight_decay "${weight_decay}" \
      --grad_clip "${grad_clip}" \
      --seed "${random_seed}" \
      --des Exp \
      "${gpu_args[@]}"
  done
done
