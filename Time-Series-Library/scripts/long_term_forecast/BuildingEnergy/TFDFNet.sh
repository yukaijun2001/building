#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tsl_dir=$(cd "${script_dir}/../../.." && pwd)
project_dir=$(cd "${tsl_dir}/.." && pwd)

model_name=TFDFNet
root_path_name=${ROOT_PATH:-/home/ykj/build/data}
target_col_name=${TARGET:-electricity}

seq_lens=${SEQ_LENS:-${SEQ_LEN:-"720"}}
label_len=${LABEL_LEN:-576}
kernel_size=${KERNEL_SIZE:-25}
dropout=${DROPOUT:-0.2}

batch_size=${BATCH_SIZE:-256}
num_workers=${NUM_WORKERS:-0}
train_epochs=${EPOCHS:-30}
patience=${PATIENCE:-3}
learning_rate=${LR:-0.0001}
device=${DEVICE:-auto}

shopt -s nullglob
csv_files=("${root_path_name}"/*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
  echo "No CSV files found in ${root_path_name}" >&2
  echo "Set ROOT_PATH=/path/to/data if your dataset lives elsewhere." >&2
  exit 1
fi

cd "${tsl_dir}"

python_bin=${PYTHON:-python}
if [ -z "${PYTHON:-}" ] && ! "${python_bin}" - <<'PY' >/dev/null 2>&1
import numpy
import pandas
import torch
PY
then
  if [ -x "${HOME}/miniconda3/envs/time_series/bin/python" ]; then
    python_bin="${HOME}/miniconda3/envs/time_series/bin/python"
  fi
fi

gpu_args=()
if [ "${device}" = "cpu" ]; then
  gpu_args+=(--no_use_gpu)
fi

for data_file in "${csv_files[@]}"
do
  data_path_name=$(basename "${data_file}")
  data_name="${data_path_name%.csv}"
  enc_in=$(DATA_FILE="${data_file}" TARGET_COL="${target_col_name}" "${python_bin}" - <<'PY'
import csv
import os

path = os.environ["DATA_FILE"]
target = os.environ["TARGET_COL"]
with open(path, newline="", encoding="utf-8") as f:
    header = next(csv.reader(f))
time_cols = {"date", "timestamp"}
cols = [c for c in header if c not in time_cols and c not in {"precipDepth1HR", "precipDepth6HR"}]
if target not in cols:
    raise SystemExit(f"{path} does not contain target column {target!r}")
print(len(cols))
PY
)

  for seq_len in ${seq_lens}
  do
    for pred_len in ${PRED_LENS:-24 48 72 96}
    do
      model_id_name=${model_name}_${data_name}_${seq_len}_${pred_len}
      echo "===== Running ${model_name} on ${data_name}: seq_len=${seq_len}, pred_len=${pred_len} ====="
      "${python_bin}" -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${root_path_name}/" \
        --data_path "${data_path_name}" \
        --model_id "${model_id_name}" \
        --model "${model_name}" \
        --data building_data \
        --features MS \
        --target "${target_col_name}" \
        --freq h \
        --seq_len "${seq_len}" \
        --label_len "${label_len}" \
        --pred_len "${pred_len}" \
        --enc_in "${enc_in}" \
        --dec_in "${enc_in}" \
        --c_out 1 \
        --in_channels "${enc_in}" \
        --out_channels 1 \
        --kernel_size "${kernel_size}" \
        --dropout "${dropout}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --train_epochs "${train_epochs}" \
        --patience "${patience}" \
        --learning_rate "${learning_rate}" \
        --des exp \
        --itr 1 \
        "${gpu_args[@]}"
    done
  done
done
