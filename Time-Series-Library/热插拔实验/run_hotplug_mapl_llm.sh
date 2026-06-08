#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/.." && pwd)

root_path_name=${ROOT_PATH:-/home/ykj/build/jie_project/dataset/building}
mapl_mark_root=${MAPL_MARK_ROOT:-${root_path_name}}
mapl_project=${MAPL_PROJECT:-/home/ykj/build/jie_project}
llm_ckp_dir=${LLM_CKP_DIR:-/home/ykj/build/llama_model}
data_path_name=${DATA_PATH:-Hog_assembly_Colette.csv}
target_col_name=${TARGET:-electricity}
random_seed=${SEED:-42}
seq_lens=${SEQ_LENS:-${SEQ_LEN:-"720"}}
pred_lens=${PRED_LENS:-"24 48 72 96"}

label_len=${LABEL_LEN:-24}
patch_len=${PATCH_LEN:-24}
patch_stride=${PATCH_STRIDE:-$((patch_len / 2))}
top_k_patches=${TOP_K_PATCHES:-8}
hidden_size=${HIDDEN_SIZE:-8}
lstm_layers=${LSTM_LAYERS:-1}
model_dim=${MODEL_DIM:-128}
fft_keep_ratio=${FFT_KEEP_RATIO:-0.5}

batch_size=${BATCH_SIZE:-16}
num_workers=${NUM_WORKERS:-0}
train_epochs=${EPOCHS:-30}
patience=${PATIENCE:-3}
learning_rate=${LR:-0.0001}
weight_decay=${WEIGHT_DECAY:-0.00001}
grad_clip=${GRAD_CLIP:-1.0}
disable_residual=${DISABLE_RESIDUAL:-0}

if [ ! -f "${root_path_name}/${data_path_name}" ]; then
  echo "CSV file not found: ${root_path_name}/${data_path_name}" >&2
  exit 1
fi

mark_name="${data_path_name%.csv}.pt"
if [ ! -f "${mapl_mark_root}/${mark_name}" ]; then
  echo "MaPL mark file not found: ${mapl_mark_root}/${mark_name}" >&2
  exit 1
fi

cd "${tsl_dir}"

residual_args=()
if [ "${disable_residual}" = "1" ]; then
  residual_args+=(--disable_residual)
fi

for seq_len in ${seq_lens}
do
  for pred_len in ${pred_lens}
  do
    model_id_name=HotPlug_MaPL_LLM_${seq_len}_${pred_len}
    "${PYTHON:-python}" -u "热插拔实验/hotplug_run.py" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${root_path_name}" \
      --data_path "${data_path_name}" \
      --source_output_dir "${tsl_dir}/热插拔实验/outputs" \
      --model_id "${model_id_name}" \
      --baseline MaPL_LLM \
      --mapl_mark_root "${mapl_mark_root}" \
      --mapl_project "${mapl_project}" \
      --llm_ckp_dir "${llm_ckp_dir}" \
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
      --des HotPlugMaPL \
      "${residual_args[@]}"
  done
done
