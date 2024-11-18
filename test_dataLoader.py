import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data_provider.data_loader import Dataset_Cluster_Trace_2018
import argparse
# 示例使用
args = argparse.Namespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id='wandb_BAYES_25S_TF',
        model='DATATEST',
        data= 'CT2018',
        root_path='/home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByCoreTimePreFrame/dataSampleFrame25s/statisiticByCoreTimePreFrame/',
        data_path='task_type1_CTPF_8640_6912_date.csv',
        features='S',
        target='count',
        freq='s',
        checkpoints='./checkpoints/',
        seq_len=8,
        label_len=0,
        pred_len=4,
        seasonal_patterns='Monthly',
        inverse=True,   #是否对数据进行还原
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=8,
        num_kernels=10,   #卷积核的数量
        enc_in=1,     #数据的特征维度
        dec_in=1,
        c_out=1,
        d_model=512,   #卷积层输出的通道数
        n_heads=8,
        e_layers=2,   #timesBlock的数量
        d_layers=1,
        d_ff = 2048,     #卷积层输出的通道数
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        channel_independence=1,
        decomp_method='moving_avg',
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        num_workers=10,
        itr=1,
        train_epochs=100,
        batch_size=128,
        patience=3,
        learning_rate=0.0001,
        des='test',
        loss='MSE',
        lradj='type1',    #学习率调整方式 type1砍半, type2按字典调整
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices='0,1,2,3',
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len = 16,  # TimeXer
        p_arima = 1, # arima
        d_arima = 1, # arima
        q_arima = 1 # arima
    )
args.use_gpu = True
print(torch.cuda.is_available())

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
# root_path = './data'
dataset = Dataset_Cluster_Trace_2018(args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag='train',
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=1,
            freq=args.freq,
            seasonal_patterns=args.seasonal_patterns,
            scale= False)

# 创建 DataLoader
dataloader = DataLoader(dataset,batch_size=4, shuffle=True)
count = 0
# 迭代数据
for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
    print("batch_x shape:", batch_x.shape)
    print(batch_x)
    print("batch_y shape:", batch_y.shape)
    print(batch_y)
    count += 1
    if count == 3:

    # print("batch_y shape:", batch_y.shape)
    # print(batch_y)
    # print("batch_x_mark shape:", batch_x_mark.shape)
    # print(batch_x_mark)
    # print("batch_y_mark shape:", batch_y_mark.shape)
    # print(batch_y_mark)
        break  # 只打印第一个批次的数据
