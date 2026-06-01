model_name=SMETimes_Llama

for pred_len in 12 24 36 48 60 72 84 96 108 120 132 144 156 168
do

  label_len=$((672 - pred_len))

  python -u preprocess.py \
    --seq_len 672 \
    --label_len $label_len \
    --pred_len $pred_len \

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity_672_None \
    --model $model_name \
    --data electricity \
    --seq_len 672 \
    --label_len $label_len \
    --token_len $pred_len \
    --test_seq_len 672 \
    --test_label_len $label_len \
    --test_pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --weight_decay 0.00001 \
    --mlp_hidden_dim 1024 \
    --mlp_activation relu \
    --train_epochs 10 \
    --use_amp \
    --gpu 0 \
    --cosine \
    --tmax 10 \
    --mix_embeds \
    --drop_last
done
