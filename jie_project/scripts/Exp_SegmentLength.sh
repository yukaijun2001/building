model=Times_Llama
data=building_single
hidden_dim=1024
batch_size=256

for data_name in Hog_assembly_Colette.csv 
do
  for pred_len in 24 48 72 96 
  do
    seq_len=672
    label_len=$((seq_len - pred_len))

    # python -u preprocess.py \
    #   --dataset $data_name \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --batch_size $batch_size \

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/building/ \
      --data_path $data_name \
      --model_id segment_length \
      --model $model \
      --data $data \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --test_seq_len $seq_len \
      --test_label_len $label_len \
      --test_pred_len $pred_len \
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
done
