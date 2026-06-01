data_name=Hog_parking_Jean.csv
model=SMETimes_Llama
data=building_single
batch_size=128

python -u preprocess.py \
    --dataset $data_name \
    --seq_len 672 \
    --label_len 648 \
    --pred_len 24 \
    --batch_size $batch_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/building/ \
  --data_path  $data_name\
  --model_id Electricity_672_24 \
  --model $model \
  --data $data \
  --seq_len 672 \
  --label_len 648 \
  --pred_len 24 \
  --test_seq_len 672 \
  --test_label_len 648 \
  --test_pred_len 24 \
  --batch_size $batch_size \
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_layers 2 \
  --mlp_activation relu \
  --train_epochs 30 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --num_experts 1 \
  --mix_embeds \
  --drop_last


#for test_pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/building/ \
#  --data_path $data_name \
#  --model_id Electricity_672_96 \
#  --model $model \
#  --data $data \
#  --seq_len 672 \
#  --label_len 576 \
#  --token_len 96 \
#  --test_seq_len 672 \
#  --test_label_len 576 \
#  --test_pred_len $test_pred_len \
#  --learning_rate 0.0001 \
#  --weight_decay 0.00001 \
#  --mlp_hidden_dim 1024 \
#  --mlp_hidden_layers 2 \
#  --mlp_activation relu \
#  --train_epochs 10 \
#  --use_amp \
#  --gpu 0 \
#  --cosine \
#  --tmax 10 \
#  --num_experts 1 \
#  --mix_embeds \
#  --drop_last \
#  --test_dir long_term_forecast_Electricity_672_96_${model}_${data}_sl672_ll576_tl96_lr0.0001_bt256_wd1e-05_hd1024_hl2_cosTrue_mixTrue_test_0
#done
