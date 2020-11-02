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

class SimpleLSTM(nn.Module):
    def __init__(self, num_outputs, add_softmax, config):
        super(SimpleLSTM, self).__init__()
        self.device = config['device']
        self.output_transformation = nn.Softmax(dim=1) if add_softmax else lambda x: x

        self.num_mfccs = config['num_mfccs']
        self.num_lstm_hidden = config['num_lstm_hidden']
        self.num_lstm_layers = config['num_lstm_layers']
        self.num_outputs = num_outputs
        
        self.lstm = nn.LSTM(
            self.num_mfccs, 
            self.num_lstm_hidden, 
            self.num_lstm_layers, 
            batch_first=True
        )
        self.fc1 = nn.Linear(self.num_lstm_hidden, self.num_outputs)
        
    def get_initial_state(self, batch_size):
        return torch.zeros(
            self.num_lstm_layers, 
            batch_size, 
            self.num_lstm_hidden, 
            device=self.device
        )

    def forward(self, x, sequences_lengths):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, 
            sequences_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        x, _ = self.lstm(x, (h0, c0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.fc1(x[:, -1, :])
        return self.output_transformation(x)

    def test_forward(self, x, sequences_lengths, idx):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, 
            sequences_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        x, _ = self.lstm(x, (h0, c0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:, idx, :]
        x = self.fc1(x)
        return self.output_transformation(x)

class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_outputs, add_softmax, config):
        super(SimpleCNNLSTM, self).__init__()
        self.device = config['device']
        self.output_transformation = nn.Softmax(dim=1) if add_softmax else lambda x: x

        self.num_mfccs = config['num_mfccs']
        self.num_lstm_hidden = config['num_lstm_hidden']
        self.num_lstm_layers = config['num_lstm_layers']
        self.num_outputs = num_outputs
        
        self.cnn = nn.Conv1d(self.num_mfccs, 50, kernel_size = 11, stride = 1, padding=5)
        self.lstm = nn.LSTM(50, self.num_lstm_hidden, self.num_lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(self.num_lstm_hidden, self.num_outputs)
        
    def get_initial_state(self, batch_size):
        return torch.zeros(self.num_lstm_layers, batch_size, self.num_lstm_hidden, device=self.device)

    def forward(self, x, sequences_lengths):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, 
            sequences_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        x, _ = self.lstm(x, (h0, c0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.fc1(x[:, -1, :])
        return self.output_transformation(x)

    def test_forward(self, x, sequences_lengths, idx):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, 
            sequences_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        x, _ = self.lstm(x, (h0, c0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:, idx, :]
        x = self.fc1(x)
        return self.output_transformation(x)



class BiggerCNNLSTM(nn.Module):
    def __init__(self, num_outputs, add_softmax, config):
        super(BiggerCNNLSTM, self).__init__()
        self.device = config['device']
        self.output_transformation = nn.Softmax(dim=1) if add_softmax else lambda x: x

        self.num_mfccs = config['num_mfccs']
        self.num_lstm_hidden = config['num_lstm_hidden']
        self.num_lstm_layers = config['num_lstm_layers']
        self.num_outputs = num_outputs
        
        self.cnn1 = nn.Conv1d(self.num_mfccs, 512, kernel_size = 11, stride = 1)
        self.cnn2 = nn.Conv1d(512, 512, kernel_size = 9, stride = 1)
        self.cnn3 = nn.Conv1d(512, 256, kernel_size = 5, stride = 1)
        self.lstm = nn.LSTM(256, self.num_lstm_hidden, self.num_lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(self.num_lstm_hidden, 256)
        self.fc2 = nn.Linear(256, self.num_outputs)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(2)
        
    def get_initial_state(self, batch_size):
        return torch.zeros(self.num_lstm_layers, batch_size, self.num_lstm_hidden, device=self.device)

    def forward(self, x, sequences_lengths):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        # x = torch.nn.utils.rnn.pack_padded_sequence(
        #     x, 
        #     sequences_lengths, 
        #     batch_first=True, 
        #     enforce_sorted=False
        # )
        x, _ = self.lstm(x, (h0, c0))
        # x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return self.output_transformation(x)

    def test_forward(self, x, sequences_lengths, idx):
        h0 = self.get_initial_state(x.shape[0])
        c0 = self.get_initial_state(x.shape[0])
        
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        # x = torch.nn.utils.rnn.pack_padded_sequence(
        #     x, 
        #     sequences_lengths, 
        #     batch_first=True, 
        #     enforce_sorted=False
        # )
        x, _ = self.lstm(x, (h0, c0))
        # x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        print(x.shape)
        x = x[:, idx, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.output_transformation(x)

class SimpleRNN(nn.Module):
    def __init__(self, num_outputs, add_softmax, config):
        super(SimpleRNN, self).__init__()
        self.device = config['device']
        self.output_transformation = nn.Softmax(dim=1) if add_softmax else lambda x: x

        self.num_mfccs = config['num_mfccs']
        self.num_lstm_hidden = config['num_lstm_hidden']
        self.num_lstm_layers = config['num_lstm_layers']
        self.num_outputs = num_outputs
        
        self.lstm = nn.RNN(
            self.num_mfccs, 
            self.num_lstm_hidden, 
            self.num_lstm_layers, 
            batch_first=True
        )
        self.fc1 = nn.Linear(self.num_lstm_hidden, self.num_outputs)
        
    def get_initial_state(self, batch_size):
        return torch.zeros(self.num_lstm_layers, batch_size, self.num_lstm_hidden, device=self.device)

    def forward(self, x, sequences_lengths):
        h0 = self.get_initial_state(x.shape[0])
        x, _ = self.lstm(x, h0)
        x = self.fc1(x[:, -1, :])
        return self.output_transformation(x)