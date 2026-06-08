#!/usr/bin/env bash
# NLinear TimeXer iTransformer TimeMixer PatchTST
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/.." && pwd)
project_dir=$(cd "${tsl_dir}/.." && pwd)

root_path_name=${ROOT_PATH:-${project_dir}/data}
target_col_name=${TARGET:-electricity}
random_seed=${SEED:-42}
baselines=${BASELINES:-" NLinear "}
mapl_mark_root=${MAPL_MARK_ROOT:-/home/ykj/build/jie_project/dataset/building}
mapl_project=${MAPL_PROJECT:-/home/ykj/build/jie_project}
llm_ckp_dir=${LLM_CKP_DIR:-/home/ykj/build/llama_model}
seq_lens=${SEQ_LENS:-${SEQ_LEN:-"672"}}
pred_lens=${PRED_LENS:-"24"}

label_len=${LABEL_LEN:-24}
patch_len=${PATCH_LEN:-24}
patch_stride=${PATCH_STRIDE:-$((patch_len / 2))}
top_k_patches=${TOP_K_PATCHES:-8}
hidden_size=${HIDDEN_SIZE:-8}
lstm_layers=${LSTM_LAYERS:-1}
model_dim=${MODEL_DIM:-128}
fft_keep_ratio=${FFT_KEEP_RATIO:-0.5}

baseline_patch_len=${BASELINE_PATCH_LEN:-16}
baseline_patch_stride=${BASELINE_PATCH_STRIDE:-8}
baseline_d_model=${BASELINE_D_MODEL:-128}
baseline_n_heads=${BASELINE_N_HEADS:-4}
baseline_e_layers=${BASELINE_E_LAYERS:-2}
baseline_d_ff=${BASELINE_D_FF:-256}

batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-0}
train_epochs=${EPOCHS:-1000}
patience=${PATIENCE:-3}
learning_rate=${LR:-0.0001}
weight_decay=${WEIGHT_DECAY:-0.0001}
grad_clip=${GRAD_CLIP:-1.0}
device=${DEVICE:-auto}
disable_residual=${DISABLE_RESIDUAL:-0}

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
residual_args=()
if [ "${disable_residual}" = "1" ]; then
  residual_args+=(--disable_residual)
fi

for baseline in ${baselines}
do
  for seq_len in ${seq_lens}
  do
    for pred_len in ${pred_lens}
    do
      model_id_name=HotPlug_${baseline}_${seq_len}_${pred_len}
      "${PYTHON:-python}" -u "热插拔实验/hotplug_run.py" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${root_path_name}" \
        --source_output_dir "${tsl_dir}/热插拔实验/outputs" \
        --model_id "${model_id_name}" \
        --baseline "${baseline}" \
        --mapl_mark_root "${mapl_mark_root}" \
        --mapl_project "${mapl_project}" \
        --llm_ckp_dir "${llm_ckp_dir}" \
        --target "${target_col_name}" \
        --seq_len "${seq_len}" \
        --label_len "${label_len}" \
        --pred_len "${pred_len}" \
        --baseline_patch_len "${baseline_patch_len}" \
        --baseline_patch_stride "${baseline_patch_stride}" \
        --baseline_d_model "${baseline_d_model}" \
        --baseline_n_heads "${baseline_n_heads}" \
        --baseline_e_layers "${baseline_e_layers}" \
        --baseline_d_ff "${baseline_d_ff}" \
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
        --des HotPlug \
        "${residual_args[@]}" \
        "${gpu_args[@]}"
    done
  done
done
