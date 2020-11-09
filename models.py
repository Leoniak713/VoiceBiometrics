import os

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

class BiggerCNNLSTM(nn.Module):
    def __init__(self, num_outputs, add_softmax, config):
        super(BiggerCNNLSTM, self).__init__()
        self.config = config
        self.device = config['device']
        self.output_transformation = nn.Softmax(dim=1) if add_softmax else lambda x: x

        self.num_mfccs = config['preprocessing']['n_mfcc']
        self.num_outputs = num_outputs
        
        self.identity = nn.Identity()
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(2)

        conv_layers_input_channels = [self.num_mfccs] + \
            [layer_args['out_channels'] for (layer_args, _) in config['conv_layers'][:-1]]

        self.conv_layers = nn.ModuleList()
        for in_channels, (layer_args, sequence_args) in zip(conv_layers_input_channels, config['conv_layers']):
            self.conv_layers.append(
                nn.Sequential(
                        nn.Conv1d(in_channels=in_channels, **layer_args),
                        nn.BatchNorm1d(layer_args['out_channels']) \
                            if sequence_args['batch_norm'] else self.identity,
                        self.relu,
                        self.pooling
                        
                )
            )

        self.lstm = nn.LSTM(
            config['conv_layers'][-1][0]['out_channels'], 
            batch_first=True,
            **config['lstm'][0]
            )
        self.batch_norm_lstm = nn.BatchNorm1d(config['lstm'][0]['hidden_size']) \
            if config['lstm'][1]['batch_norm'] else self.identity

        fc_layers_input_channels = [config['lstm'][0]['hidden_size']] + \
            [out_features for (out_features, _, _) in config['fc_layers'][:-1]]
        self.fc_layers = nn.ModuleList()
        for in_features, (out_features, batch_norm, dropout) in zip(fc_layers_input_channels, config['fc_layers']):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features) if batch_norm else self.identity,
                    self.relu,
                    nn.Dropout(dropout)
                )
            )
        self.output_layer = nn.Linear(config['fc_layers'][-1][0], self.num_outputs)
        
    def get_initial_state(self, batch_size):
        return torch.zeros(
            self.config['lstm'][0]['num_layers'], 
            batch_size, 
            self.config['lstm'][0]['hidden_size'], 
            device=self.device
            )

    def forward(self, x, lstm_position=-1):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        
        x = x.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, lstm_position, :]
        x = self.batch_norm_lstm(x)
        x = self.relu(x)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.output_layer(x)
        return self.output_transformation(x)