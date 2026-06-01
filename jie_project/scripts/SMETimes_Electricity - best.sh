#!/usr/bin/env bash
set -euo pipefail

data_name=Hog_assembly_Colette.csv
model=Times_MIMO
data=building_multi
seq_len=672
pred_len=96
label_len=$((seq_len - pred_len))
batch_size=256
learning_rate=0.0001
weight_decay=0.00001
mlp_hidden_dim=1024
mlp_hidden_layers=2
des=test
itr=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/building/ \
  --data_path $data_name \
  --model_id Electricity_672_96 \
  --model $model \
  --data $data \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --test_seq_len $seq_len \
  --test_label_len $label_len \
  --test_pred_len $pred_len \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_hidden_layers $mlp_hidden_layers \
  --mlp_activation relu \
  --train_epochs 30 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --num_experts 1 \
  --mix_embeds \
  --drop_last


for test_pred_len in 24 48 72 96
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/building/ \
    --data_path $data_name \
    --model_id Electricity_672_96 \
    --model $model \
    --data $data \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_label_len $label_len \
    --test_pred_len $test_pred_len \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --mlp_hidden_dim $mlp_hidden_dim \
    --mlp_hidden_layers $mlp_hidden_layers \
    --mlp_activation relu \
    --train_epochs 10 \
    --use_amp \
    --gpu 0 \
    --cosine \
    --tmax 10 \
    --num_experts 1 \
    --mix_embeds \
    --drop_last \
    --test_dir long_term_forecast_Electricity_672_96_${model}_${data}_sl${seq_len}_ll${label_len}_tl${pred_len}_lr${learning_rate}_bt${batch_size}_wd${weight_decay}_hd${mlp_hidden_dim}_hl${mlp_hidden_layers}_cosTrue_mixTrue_${des}_${itr}
done
