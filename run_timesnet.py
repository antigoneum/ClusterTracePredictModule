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
run_project_name = "TimesNet_25CTPF_multiFuture"
run_notes = "timesnet"
run_tags = ["timesnet"]
sweep_name = "TimesNet_25CTPF_pre1_sumFeature"

# os.environ['WANDB_MODE'] = 'offline'

sweep_config = {
    # "method": "bayes",
    "method": "bayes",
    "name": f"{sweep_name}",
    "metric": {"name": "last_vali_loss", "goal": "minimize"},
    "parameters": {
        # "batch_size": {"values": [64, 128]},  
        # "learning_rate": {"values": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,0.02,0.05,0.1]},  #学习率
        "learning_rate": {"max": 0.1, "min": 0.0001, "distribution": "uniform"},
        # "learning_rate": {"values" : [0.0725]},  #学习率
        # "seq_len": {"values": [4096]},  #历史数据长度
        "seq_len": {"max": 64, "min": 8, "distribution": "int_uniform"},  #历史数据长度
        "pred_len": {"values": [1]},  #预测的长度
        "e_layers": {"max": 3, "min":1, "distribution": "int_uniform"},   
        "d_model": {"max": 128, "min": 8, "distribution": "q_log_uniform_values", "q":2},  #卷积层输入的通道数  自编码器 
        "d_layers": {"max": 3, "min": 1, "distribution": "int_uniform"}, 
        "d_ff": {"max": 128, "min": 8, "distribution": "q_log_uniform_values", "q":2},    #卷积层输出的通道数  自编码器 
        # "dropout": {"values": [0.6]}, #dropout
        "dropout": {"max": 0.9, "min": 0.1, "distribution": "uniform"}, #dropout
        "num_kernels": {"max": 10, "min": 5, "distribution": "int_uniform"},  #卷积核的数量
        # "train_epochs": {"values": [30, 50]},  #训练的轮数
        "top_k": {"max" : 12, "min": 5, "distribution": "int_uniform"} ,  #top_k
        "lradj": {"values": ["type1","cosine"]},  #学习率调整方式 type1砍半, type2按字典调整
        # "factor": {"max" : 6 , "min": 2},  #注意力因子
        # "moving_avg": {"values": list(range(3,33,2))},  #滑动平均
        "loss": {"values": ["MSE"]},  #损失函数
        # "p_arima" :{"values": [0,1,2]},
        # "d_arima" :{"values": [0,1,2]},
        # "q_arima" :{"values": [0,1,2]},
        # "seq_len": {"values": [4096]},  #历史数据长度
    }
}

sweep_id = wandb.sweep(sweep_config, project=run_project_name)

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args = argparse.Namespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id='wandb_BAYES_25S_TF',
        model='TimesNet',
        data= 'CT2018',
        root_path='/home/antigone/cluster-trace-predict/ClusterTracePredictModule/dataset/cluster_trace_2018/statisticsByCoreTimePreFrame/dataSampleFrame25s/statisiticByCoreTimePreFrame/',
        data_path='task_type1_CTPF_8640_6912_date_sumdiff.csv',
        features='MS',
        target='count',
        freq='s',
        checkpoints='./checkpoints/',
        seq_len=96,
        label_len=0,
        pred_len=1,
        seasonal_patterns='Monthly',
        inverse=True,   #是否对数据进行还原
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=10,   #卷积核的数量
        enc_in=9,     #数据的特征维度
        dec_in=9,
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
        q_arima = 1, # arima
        gru_layer = 1,
        gru_hidden_size = 128,
        gru_mlp_layers = 2,
        gru_mlp_size = 128,
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
    project=run_project_name,
    notes = run_notes,
    tags = run_tags,
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
            # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            #     args.task_name,
            #     args.model_id,
            #     args.model,
            #     args.data,
            #     args.features,
            #     args.seq_len,
            #     args.label_len,
            #     args.pred_len,
            #     args.d_model,
            #     args.n_heads,
            #     args.e_layers,
            #     args.d_layers,
            #     args.d_ff,
            #     args.expand,
            #     args.d_conv,
            #     args.factor,
            #     args.embed,
            #     args.distil,
            #     args.des, ii)
            setting = f"{args.des}_{ii}"
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.task_name,
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.expand,
        #     args.d_conv,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)
        setting = f"{args.des}_{ii}"
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=128)
