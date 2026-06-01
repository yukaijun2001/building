#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/.." && pwd)
project_dir=$(cd "${tsl_dir}/.." && pwd)

if [ "$#" -gt 0 ]; then
  model_names=("$@")
else
  read -r -a model_names <<< "${MODELS:-${MODEL:-PatchTST iTransformer TimeMixer TimeXer}}"
fi

output_root=${OUTPUT_ROOT:-${script_dir}/output}
summary_csv=${SUMMARY_CSV:-${output_root}/summary.csv}
root_path_name=${ROOT_PATH:-${project_dir}/data}
target_col_name=${TARGET:-electricity}
random_seed=${SEED:-42}

seq_len_default=${SEQ_LEN:-672}
time_xer_seq_len=${TIME_XER_SEQ_LEN:-${seq_len_default}}
label_len=${LABEL_LEN:-24}
patch_len=${PATCH_LEN:-24}

d_model=${D_MODEL:-512}
n_heads=${N_HEADS:-8}
e_layers=${E_LAYERS:-2}
d_layers=${D_LAYERS:-1}
d_ff=${D_FF:-2048}
factor=${FACTOR:-1}
activation=${ACTIVATION:-gelu}
dropout=${DROPOUT:-0.1}
top_k=${TOP_K:-5}
moving_avg=${MOVING_AVG:-25}
channel_independence=${CHANNEL_INDEPENDENCE:-0}
decomp_method=${DECOMP_METHOD:-moving_avg}
use_norm=${USE_NORM:-1}
down_sampling_layers=${DOWN_SAMPLING_LAYERS:-1}
down_sampling_window=${DOWN_SAMPLING_WINDOW:-2}
down_sampling_method=${DOWN_SAMPLING_METHOD:-avg}

batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-0}
train_epochs=${EPOCHS:-100}
patience=${PATIENCE:-3}
learning_rate=${LR:-0.0001}
device=${DEVICE:-cuda}
gpu=${GPU:-0}

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
else
  gpu_args+=(--use_gpu --gpu "${gpu}" --gpu_type "${device}")
fi

for model_name in "${model_names[@]}"; do
  seq_len="${seq_len_default}"
  if [ "${model_name}" = "TimeXer" ]; then
    seq_len="${time_xer_seq_len}"
  fi

  for data_file in "${csv_files[@]}"; do
    data_path_name=$(basename "${data_file}")
    data_name="${data_path_name%.csv}"
    enc_in=$(DATA_FILE="${data_file}" TARGET_COL="${target_col_name}" python - <<'PY'
import csv
import os

path = os.environ["DATA_FILE"]
target = os.environ["TARGET_COL"]
with open(path, newline="", encoding="utf-8") as f:
    header = next(csv.reader(f))
time_cols = {"date", "timestamp"}
cols = [c for c in header if c not in time_cols]
if target not in cols:
    raise SystemExit(f"{path} does not contain target column {target!r}")
print(len(cols))
PY
)

    for pred_len in ${PRED_LENS:-24}; do
      model_id_name=${model_name}_${data_name}_${seq_len}_${pred_len}
      run_output_dir=${output_root}/${model_id_name}
      checkpoints_dir=${output_root}/checkpoints
      echo "===== Measuring ${model_name} on ${data_name}: seq_len=${seq_len}, pred_len=${pred_len} ====="

      "${PYTHON:-python}" -u "${script_dir}/measure_params.py" \
        --project_dir "${tsl_dir}" \
        --output_dir "${run_output_dir}" \
        --summary_csv "${summary_csv}" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${root_path_name}/" \
        --data_path "${data_path_name}" \
        --model_id "${model_id_name}" \
        --model "${model_name}" \
        --data custom \
        --features MS \
        --target "${target_col_name}" \
        --checkpoints "${checkpoints_dir}/" \
        --seq_len "${seq_len}" \
        --label_len "${label_len}" \
        --pred_len "${pred_len}" \
        --enc_in "${enc_in}" \
        --dec_in "${enc_in}" \
        --c_out 1 \
        --patch_len "${patch_len}" \
        --d_model "${d_model}" \
        --n_heads "${n_heads}" \
        --e_layers "${e_layers}" \
        --d_layers "${d_layers}" \
        --d_ff "${d_ff}" \
        --factor "${factor}" \
        --activation "${activation}" \
        --dropout "${dropout}" \
        --top_k "${top_k}" \
        --moving_avg "${moving_avg}" \
        --channel_independence "${channel_independence}" \
        --decomp_method "${decomp_method}" \
        --use_norm "${use_norm}" \
        --down_sampling_layers "${down_sampling_layers}" \
        --down_sampling_window "${down_sampling_window}" \
        --down_sampling_method "${down_sampling_method}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --train_epochs "${train_epochs}" \
        --patience "${patience}" \
        --learning_rate "${learning_rate}" \
        --seed "${random_seed}" \
        --des Param \
        --itr 1 \
        "${gpu_args[@]}"
    done
  done
done

echo "All measurements finished."
echo "Summary CSV: ${summary_csv}"
echo "Per-run JSON files: ${output_root}/*/metrics.json"
