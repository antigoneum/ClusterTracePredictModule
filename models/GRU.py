import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Model(nn.Module):

    def __init__(self, configs, init = 'xavier'):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.pred_len = configs.pred_len
        self.d_layers = configs.d_layers
        self.activation = configs.activation
        self.gru = nn.GRU(input_size = self.enc_in, hidden_size = self.d_model, num_layers = self.e_layers, batch_first = True)
        if self.activation == 'relu':
            self.activation = nn.ReLU()
        elif self.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        model_list = []
        if self.d_layers == 0 :
            model_list.append(nn.Linear(self.d_model, self.pred_len))
        else:
            model_list.append(nn.Linear(self.d_model, self.d_ff))
            model_list.append(self.activation)
            for i in range(self.d_layers-1):
                model_list.append(nn.Linear(self.d_ff, self.d_ff))
                model_list.append(self.activation)
            model_list.append(nn.Linear(self.d_ff, self.pred_len))
        self.mlp = nn.ModuleList(model_list)    
    def _initialize_weights(self, init):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        if init == 'xavier':
                            nn.init.xavier_uniform_(param.data)
                        elif init == 'kaiming':
                            nn.init.kaiming_uniform_(param.data)
                    elif 'weight_hh' in name:
                        if init == 'xavier':
                            nn.init.xavier_uniform_(param.data)
                        elif init == 'kaiming':
                            nn.init.kaiming_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    def forward(self, x, x_mark, x_dec, x_mark_dec ):
        if self.pred_len == 1:
            out, h = self.gru(x)

            for layer in self.mlp:
                out = layer(out)
            return out
        else:
            raise NotImplementedError

        