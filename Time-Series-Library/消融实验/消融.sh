#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
用法:
  bash 消融实验/消融.sh no_frequency_denoising
  bash 消融实验/消融.sh no_patch_selection
  bash 消融实验/消融.sh no_gated_attention
  bash 消融实验/消融.sh all

可用消融参数:
  no_frequency_denoising  去掉 Frequency denoising 模块
  no_patch_selection      去掉 Bidirectional Patch Scoring and Selection，所有 patch 直接送入 LSTM
  no_gated_attention      去掉门控注意力机制，直接使用 LSTM 最后一个时间步状态
  all                     顺序跑上面三个消融

常用环境变量:
  ROOT_PATH=/path/to/csv_dir
  SEQ_LEN=672 或 SEQ_LENS="168 336 672"
  PRED_LENS="24 48 72 96"
  EPOCHS=1000 PATIENCE=3 BATCH_SIZE=64 DEVICE=gpu
USAGE
}

normalize_ablation() {
  local key="${1//-/_}"
  key="${key,,}"
  case "${key}" in
    all)
      printf '%s\n' "all"
      ;;
    none|baseline|full)
      printf '%s\n' "none"
      ;;
    freq|frequency|denoising|frequency_denoising|no_freq|no_frequency|no_denoising|no_frequency_denoising)
      printf '%s\n' "no_frequency_denoising"
      ;;
    patch|patch_selection|patch_scoring|bidirectional_patch|bidirectional_patch_selection|no_patch|no_patch_scoring|no_patch_selection)
      printf '%s\n' "no_patch_selection"
      ;;
    gate|gated|attention|gated_attention|no_gate|no_attention|no_gated_attention)
      printf '%s\n' "no_gated_attention"
      ;;
    *)
      return 1
      ;;
  esac
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

requested_ablation=${1:-${ABLATION:-}}
if [ -z "${requested_ablation}" ]; then
  usage >&2
  exit 1
fi

ablation=$(normalize_ablation "${requested_ablation}") || {
  echo "Unknown ablation: ${requested_ablation}" >&2
  usage >&2
  exit 1
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/.." && pwd)
project_dir=$(cd "${tsl_dir}/.." && pwd)

model_name=PatchGatedLSTM
root_path_name=${ROOT_PATH:-${project_dir}/data}
target_col_name=${TARGET:-electricity}
random_seed=${SEED:-42}

seq_lens=${SEQ_LENS:-${SEQ_LEN:-"672"}}
pred_lens=${PRED_LENS:-"24 48 72 96"}
label_len=${LABEL_LEN:-24}
patch_len=${PATCH_LEN:-24}
patch_stride=${PATCH_STRIDE:-12}
top_k_patches=${TOP_K_PATCHES:-8}
hidden_size=${HIDDEN_SIZE:-8}
lstm_layers=${LSTM_LAYERS:-1}
model_dim=${MODEL_DIM:-128}
fft_keep_ratio=${FFT_KEEP_RATIO:-0.5}

batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-0}
train_epochs=${EPOCHS:-1000}
patience=${PATIENCE:-3}
learning_rate=${LR:-0.0001}
weight_decay=${WEIGHT_DECAY:-0.0001}
grad_clip=${GRAD_CLIP:-1.0}
device=${DEVICE:-gpu}
output_root=${OUTPUT_ROOT:-${script_dir}/output}

shopt -s nullglob
csv_files=("${root_path_name}"/*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
  echo "No CSV files found in ${root_path_name}" >&2
  echo "Set ROOT_PATH=/path/to/data if your dataset lives elsewhere." >&2
  exit 1
fi

cd "${tsl_dir}"

gpu_args=()
case "${device}" in
  gpu|cuda|auto)
    ;;
  cpu)
    gpu_args+=(--no_use_gpu)
    ;;
  *)
    echo "Unknown DEVICE=${device}. Use DEVICE=gpu or DEVICE=cpu." >&2
    exit 1
    ;;
esac
if [ -n "${GPU:-}" ]; then
  gpu_args+=(--gpu "${GPU}")
fi

data_args=()
if [ -n "${DATA_PATH:-}" ]; then
  data_args+=(--data_path "${DATA_PATH}")
fi
if [ -n "${MAX_FILES:-}" ]; then
  data_args+=(--max_files "${MAX_FILES}")
fi
if [ -n "${LIMIT_WINDOWS:-}" ]; then
  data_args+=(--limit_windows "${LIMIT_WINDOWS}")
fi

if [ "${ablation}" = "all" ]; then
  ablations=(no_frequency_denoising no_patch_selection no_gated_attention)
else
  ablations=("${ablation}")
fi

for current_ablation in "${ablations[@]}"
do
  for seq_len in ${seq_lens}
  do
    for pred_len in ${pred_lens}
    do
      model_id_name=${model_name}_${current_ablation}_${seq_len}_${pred_len}
      source_dir=${output_root}/source_${model_id_name}

      "${PYTHON:-python}" -u build_run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${root_path_name}" \
        --source_output_dir "${source_dir}" \
        --model_id "${model_id_name}" \
        --model PatchGatedLSTM \
        --data BuildingEnergy \
        --features MS \
        --target "${target_col_name}" \
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
        --ablation "${current_ablation}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --train_epochs "${train_epochs}" \
        --patience "${patience}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --grad_clip "${grad_clip}" \
        --seed "${random_seed}" \
        --des "Ablation_${current_ablation}" \
        "${gpu_args[@]}" \
        "${data_args[@]}"
    done
  done
done
