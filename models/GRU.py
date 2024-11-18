import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class GRU(nn.Module):
    def __init__(self, configs, init = 'xavier'):
        super(GRU, self).__init__()
        self.gru_layer = configs.gru_layer
        self.gru_hidden_size = configs.gru_hidden_size
        self.gru_input_size = configs.enc_in
        # self.gru = nn.GRU(input_size = self.gru_input_size, hidden_size = self.gru_hidden_size, num_layers = self.gru_layer, batch_first = True)
        self.gru_mlp_layers = configs.gru_mlp_layers
        self.gru_mlp_size = configs.gru_mlp_size
        self.output_size = configs.c_out
        self.activation = configs.activation
        if self.activation == 'relu':
            self.activation = nn.ReLU()
        elif self.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        model_list = []
        self.gru = nn.GRU(input_size = self.gru_input_size, hidden_size = self.gru_hidden_size, num_layers = self.gru_layer, batch_first = True)
        if self.gru_mlp_layers <= 1 :
            model_list.append(nn.Linear(self.gru_hidden_size, self.output_size))
        elif self.gru_mlp_layers == 2:
            model_list.append(nn.Linear(self.gru_hidden_size, self.gru_mlp_size))
            model_list.append(self.activation)
            model_list.append(nn.Linear(self.gru_mlp_size, self.output_size))
        else:
            model_list.append(nn.Linear(self.gru_hidden_size, self.gru_mlp_size))
            model_list.append(self.activation)
            for i in range(self.gru_mlp_layers-2):
                model_list.append(nn.Linear(self.gru_mlp_size, self.gru_mlp_size))
                model_list.append(self.activation)
            model_list.append(nn.Linear(self.gru_mlp_size, self.output_size))
        self.mlp = nn.Sequential(*model_list)
    def forward(self, x, h = None):
        if h is None:
            out, h = self.gru(x)
            out = self.mlp(h[-1, :, :].squeeze(0))
        else:
            out, h = self.gru(x, h)
            out = self.mlp(h[-1, :, :].squeeze(0))
        return out, h
class Model(nn.Module):

    def __init__(self, configs, init = 'xavier'):
        super(Model, self).__init__()
        self.net = GRU(configs, init)
        self.configs = configs
    def forward(self, x, x_mark, x_dec, x_mark_dec):
        if self.configs.pred_len == 1:
            out, h = self.net(x)
            return out.reshape(-1, 1, self.configs.c_out)
        else :
            outs = []
            h = None
            for i in range(self.configs.pred_len):
                if i is 0:
                    out, h = self.net(x)
                else:
                    out, h = self.net(x, h)
                outs.append(out.unsqueeze(1))
                x = torch.cat([x[:,1:,:], out.unsqueeze(1)], dim=1) #[batch, step, feature]
            out = torch.cat(outs, dim=1)
            return out
    