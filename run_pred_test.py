import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import wandb
import time



def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args = argparse.Namespace(
        task_name='long_term_forecast',
        is_training=0,
        model_id='wandb_BAYES_10S',
        model='TimesNet',
        data= 'CT2018',
        root_path='/home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/test_if_ARmodel/',
        data_path='test1.csv',
        features='S',
        target='count',
        freq='s',
        checkpoints='./checkpoints/',
        seq_len=16,
        label_len=1,
        pred_len=1,
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
        d_model=54,   #卷积层输出的通道数
        n_heads=8,
        e_layers=3,   #timesBlock的数量
        d_layers=1,
        d_ff=24,     #卷积层输出的通道数
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
        train_epochs=20,
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
    )
    args.use_gpu = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    wandb.config = vars(args) 
    run = wandb.init(
    project="ClusterTracePredictModule_sweep_lr_BAYES_testAR",
    notes = "timesnet",
    tags = ["timesnet","lr","dateEmbed","BAYES","testAR"],
    config = wandb.config
)
    args = argparse.Namespace(**wandb.config)
    args.des = run.id
    print('Args in experiment:')
    print_args(args)
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test("test_if_ARmodel", test=1)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()