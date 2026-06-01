#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/.." && pwd)
project_dir=$(cd "${tsl_dir}/.." && pwd)

model_name=PatchGatedLSTM
root_path_name=${ROOT_PATH:-${project_dir}/data}
target_col_name=${TARGET:-electricity}
random_seed=${SEED:-42}

seq_lens=${SEQ_LENS:-${SEQ_LEN:-"672"}}
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
device=${DEVICE:-auto}
gpu=${GPU:-0}

output_root=${OUTPUT_ROOT:-${script_dir}/output/patchgatedlstm参数}
summary_csv=${SUMMARY_CSV:-${output_root}/summary.csv}
checkpoints_dir=${CHECKPOINTS_DIR:-${output_root}/checkpoints}

mkdir -p "${output_root}"
rm -f "${summary_csv}"

shopt -s nullglob
csv_files=("${root_path_name}"/*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
  echo "No CSV files found in ${root_path_name}" >&2
  echo "Set ROOT_PATH=/path/to/data if your dataset lives elsewhere." >&2
  exit 1
fi

gpu_args=()
if [ "${device}" = "cpu" ]; then
  gpu_args+=(--no_use_gpu)
elif [ "${device}" != "auto" ]; then
  gpu_args+=(--use_gpu --gpu "${gpu}" --gpu_type "${device}")
fi

for seq_len in ${seq_lens}; do
  for pred_len in ${PRED_LENS:-24}; do
    model_id_name=${model_name}_${seq_len}_${pred_len}
    run_output_dir=${output_root}/${model_id_name}
    echo "===== Measuring ${model_id_name} ====="

    "${PYTHON:-python}" -u "${script_dir}/measure_patchgatedlstm_params.py" \
      --project_dir "${tsl_dir}" \
      --output_dir "${run_output_dir}" \
      --summary_csv "${summary_csv}" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${root_path_name}" \
      --checkpoints "${checkpoints_dir}/" \
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
      --batch_size "${batch_size}" \
      --num_workers "${num_workers}" \
      --train_epochs "${train_epochs}" \
      --patience "${patience}" \
      --learning_rate "${learning_rate}" \
      --weight_decay "${weight_decay}" \
      --grad_clip "${grad_clip}" \
      --seed "${random_seed}" \
      --des Param \
      "${gpu_args[@]}"
  done
done

echo "All measurements finished."
echo "Summary CSV: ${summary_csv}"
echo "Per-run JSON files: ${output_root}/*/metrics.json"
