import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import wandb
from multiprocessing import Pool
from statsmodels.tsa.arima.model import ARIMA

# 生成示例时间序列数据
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.p_arima = configs.p_arima  # ARIMA(p, d, q)中的p,表示自回归项阶数
        self.d_arima = configs.d_arima  # ARIMA(p, d, q)中的d,表示时间序列需要几阶差分平稳
        self.q_arima = configs.q_arima  # ARIMA(p, d, q)中的q,表示移动平均项阶数
        self.seq_len = configs.seq_len
        self.pre_len = configs.pred_len
    def fit_and_forecast(self, x):
        Model = ARIMA(x, order=(self.p_arima, self.d_arima, self.q_arima))
        Model_fit = Model.fit()
        y_pred = Model_fit.forecast(steps=self.pre_len)
        return y_pred
    def forward(self, x):
        # 拟合ARIMA模型
        assert x.shape[2] == 1
        x = x.numpy().squeeze()
        y_preds = []
        

        with Pool() as pool:
            y_preds = pool.map(self.fit_and_forecast, [x[i] for i in range(x.shape[0])])

        # for i in range(x.shape[0]):
        #     model = ARIMA(x[i], order=(self.p_arima, self.d_arima, self.q_arima))
        #     print(f"{i+1}/{x.shape[0]}", end="\r")
        #     model_fit = model.fit()
        #     # 预测未来n个时间步
        #     y_pred = model_fit.forecast(steps=self.pre_len)
        #     y_preds.append(y_pred)
        y_preds = np.vstack(y_preds).reshape(x.shape[0], self.pre_len, 1)
        print(y_preds.shape)
        return y_preds
        
