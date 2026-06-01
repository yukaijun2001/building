#!/usr/bin/env bash

set -euo pipefail

# 模型名称。可选 PatchGatedLSTM / PatchGatedSegRNN。
# 可覆盖：MODEL=PatchGatedSegRNN bash code/scripts/building_energy.sh
model_name=${MODEL:-PatchGatedLSTM}

# 数据目录。脚本从这里读取所有建筑 CSV：./data/*.csv。
root_path_name=./data

# 预测目标列。当前建筑能耗数据里的目标列是 electricity。
target_col_name=electricity

# 随机种子。可运行时覆盖：SEED=2024 bash code/scripts/building_energy.sh
random_seed=${SEED:-42}

# 输入历史窗口长度。168 表示用过去 168 小时，也就是 7 天历史。
seq_len=${SEQ_LEN:-168}

# 每个 patch 的长度。24 表示一个 patch 覆盖 24 小时。
patch_len=${PATCH_LEN:-24}

# SegRNN 的分段长度。24 表示以一天为一个 segment。
seg_len=${SEG_LEN:-24}

# patch 滑动步长。12 表示相邻 patch 间隔 12 小时，会有 12 小时重叠。
patch_stride=${PATCH_STRIDE:-12}

# 选择的重要 patch 数量。模型先给所有 patch 打分，再选 top-k。
top_k_patches=${TOP_K_PATCHES:-8}

# LSTM 隐藏层维度。越大表达能力越强，但训练更慢、更容易过拟合。
#目前来看，越低效果越好
hidden_size=${HIDDEN_SIZE:-8}

# LSTM 编码器层数。多层训练可运行：LSTM_LAYERS=3 bash code/scripts/building_energy.sh
lstm_layers=${LSTM_LAYERS:-1}

# patch token 投影后的特征维度，送入最终 LSTM 编码器前使用。
model_dim=${MODEL_DIM:-128}

# 批大小。显存/内存不够时调小，例如 BATCH_SIZE=32。
batch_size=${BATCH_SIZE:-64}

# DataLoader 进程数。一般 0 就够；磁盘和 CPU 足够时可设 NUM_WORKERS=2/4。
num_workers=${NUM_WORKERS:-0}

# 训练轮数。每栋建筑都会最多训练这么多轮，早停可能提前结束。
train_epochs=${EPOCHS:-100}

# 早停耐心。默认 3，避免验证集短期震荡导致过早停止。
patience=${PATIENCE:-3}

# 学习率。默认 0.001，训练不稳定可调小，例如 LR=0.0005。
learning_rate=${LR:-0.0001}

# 设备。auto 表示有 CUDA 就用 GPU，否则用 CPU；也可指定 DEVICE=cpu 或 DEVICE=cuda。
device=${DEVICE:-cuda}

# 预测长度列表。默认分别预测未来 24/48/96 小时。
# 可覆盖：PRED_LENS="24" bash code/scripts/building_energy.sh
for pred_len in ${PRED_LENS:-24}
do
    # 每个预测长度单独建一个输出目录，避免结果互相覆盖。
    model_id_name=${model_name}_${seq_len}_${pred_len}
    python -u code/run_experiment.py \
      --data-dir $root_path_name \
      --output-dir code/outputs/$model_id_name \
      --model $model_name \
      --target-col $target_col_name \
      --seq-len $seq_len \
      --pred-len $pred_len \
      --seg-len $seg_len \
      --patch-len $patch_len \
      --patch-stride $patch_stride \
      --top-k-patches $top_k_patches \
      --hidden-size $hidden_size \
      --lstm-layers $lstm_layers \
      --model-dim $model_dim \
      --epochs $train_epochs \
      --patience $patience \
      --batch-size $batch_size \
      --num-workers $num_workers \
      --lr $learning_rate \
      --seed $random_seed \
      --device $device
done
