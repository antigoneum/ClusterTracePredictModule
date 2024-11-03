model_name=TimesNet


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByStartTime/ \
  --data_path task_type1_start_time_10_60_date.csv \
  --model_id t1_ST_topk_5_onestep_lookback512 \
  --model $model_name \
  --data CT2018 \
  --features S \
  --seq_len  128\
  --label_len 64 \
  --pred_len 1 \
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


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByStartTime/ \
#   --data_path task_type1_start_time_10_60_date.csv \
#   --model_id t1_ST_topk_5_onestep_lookback256 \
#   --model $model_name \
#   --data CT2018 \
#   --features S \
#   --seq_len  256\
#   --label_len 128 \
#   --pred_len 1 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --target 'count' \
#   --freq 's' \
#   --batch_size 64 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByStartTime/ \
#   --data_path task_type1_start_time_10_60_date.csv \
#   --model_id t1_ST_topk_5_onestep_lookback64 \
#   --model $model_name \
#   --data CT2018 \
#   --features S \
#   --seq_len  64\
#   --label_len 32 \
#   --pred_len 1 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --target 'count' \
#   --freq 's' \
#   --batch_size 64 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByStartTime/ \
#   --data_path task_type1_start_time_10_60_date.csv \
#   --model_id t1_ST_topk_5_onestep_lookback32 \
#   --model $model_name \
#   --data CT2018 \
#   --features S \
#   --seq_len  32\
#   --label_len 16 \
#   --pred_len 1 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --target 'count' \
#   --freq 's' \
#   --batch_size 64 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByStartTime/ \
#   --data_path task_type1_start_time_10_60_date.csv \
#   --model_id t1_ST_topk_5_onestep_lookback16 \
#   --model $model_name \
#   --data CT2018 \
#   --features S \
#   --seq_len  16\
#   --label_len 8 \
#   --pred_len 1 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --target 'count' \
#   --freq 's' \
#   --batch_size 64 \
