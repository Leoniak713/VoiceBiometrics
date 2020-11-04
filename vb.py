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

from models import SimpleLSTM, SimpleCNNLSTM, SimpleRNN
from data import VoxDataset, InputNormalizer, DataBuilder

class Trainer:
    def __init__(self, config):
        self.config = config
        if self.config['loss'] == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
            onehot_encoding = False
            add_softmax = False
        elif self.config['loss'] == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
            onehot_encoding = True
            add_softmax = True
        else:
            raise BaseException('Invalid loss')
        self.device = config['device']
        #self.data = VoxDataset(config['data_path'], onehot_encoding=onehot_encoding, n_mfcc=config['num_mfccs'])
        self.data_builder = DataBuilder(
            config['data_path'], 
            onehot_encoding=onehot_encoding, 
            n_mfcc=config['num_mfccs'], 
            num_folds=5,
            cache_dir=config['cache_dir'],
            overwrite_cache = config['overwrite_cache']
            )
        self.data_train, self.data_val = self.data_builder[0]

        self.input_normalizer = InputNormalizer(self.data_train)
        self.dataloader_train = DataLoader(self.data_train, 
                                     batch_size=self.config['batch_size'], 
                                     shuffle=True, 
                                     num_workers=self.config['batch_size'], 
                                     collate_fn=self.input_normalizer.collate_fn)
        self.dataloader_val = DataLoader(self.data_val, 
                                     batch_size=self.config['batch_size'], 
                                     shuffle=True, 
                                     num_workers=self.config['batch_size'], 
                                     collate_fn=self.input_normalizer.collate_fn)
        self.net = config['network'](self.data_train.num_classes, add_softmax, self.config).to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config['learning_rate'])
        
    def train(self):
        self.net.train()
        for epoch in range(self.config['num_epochs']):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (batch, label, sequences_lengths) in enumerate(self.dataloader_train):
                batch = batch.to(self.device)
                label = label.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(batch, sequences_lengths)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            print(running_loss)

        print('Finished Training')
        
    def validate(self, validate_on_train=False, verbose=False):
        if validate_on_train:
            dataloader = self.dataloader_train
            num_records = self.data_train
        else:
            dataloader = self.dataloader_val
            num_records = self.data_val

        self.net.eval()
        accurate_predictions = 0
        with torch.no_grad():
            for i, (batch, label, sequences_lengths) in enumerate(dataloader):
                batch = batch.to(self.device)
                outputs = self.net(batch, sequences_lengths).cpu()
                predictions = np.argmax(outputs, axis=1)
                accurate_predictions += sum(predictions==label).item()
                if verbose:
                    print(outputs)
                    print(label)
        print(f"""Finished Evaluation: {accurate_predictions}/{len(num_records)} 
            ({round(accurate_predictions/len(num_records)*100, 2)}%) accurate""")