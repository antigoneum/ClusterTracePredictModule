model_name=TimesNet


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByStartTime/ \
  --data_path task_type1_start_time_10_60_date.csv \
  --model_id t1_ST_topk_5 \
  --model $model_name \
  --data CT2018 \
  --features S \
  --seq_len  512\
  --label_len 256 \
  --pred_len 512 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --target 'count' \
  --freq 's' \
  --batch_size 64 \

