model_id=ablation
model=Times_Llama_Ablation
data=building_single
hidden_dim=1024
batch_size=256

for data_name in Hog_parking_Jean.csv
do
  python -u preprocess.py \
    --dataset $data_name \
    --seq_len 672 \
    --label_len 648 \
    --pred_len 24 \
    --batch_size $batch_size \

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/building/ \
    --data_path  $data_name\
    --model_id $model_id \
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
    --mlp_hidden_dim $hidden_dim \
    --mlp_hidden_layers 2 \
    --mlp_activation relu \
    --train_epochs 30 \
    --use_amp \
    --gpu 0 \
    --cosine \
    --tmax 10 \
    --drop_last
done