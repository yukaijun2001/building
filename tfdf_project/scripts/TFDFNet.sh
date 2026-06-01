for dataset in Hog_assembly_Colette.csv 
do
for pred_len in 96
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset \
  --model_id Electricity_672_96 \
  --model TFDFNet \
  --data building_data \
  --features MS \
  --seq_len 672 \
  --label_len 576 \
  --pred_len $pred_len \
  --batch_size 256 \
  --des 'exp' \
  --target electricity \
  --train_epochs 30 \
  --itr 1
done
done