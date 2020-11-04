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
            self.add_softmax = False
        elif self.config['loss'] == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
            onehot_encoding = True
            self.add_softmax = True
        else:
            raise BaseException('Invalid loss')
        self.device = config['device']
        self.data_builder = DataBuilder(
            config['data_path'], 
            onehot_encoding = onehot_encoding, 
            n_mfcc = config['num_mfccs'], 
            num_folds = 5,
            cache_dir = config['cache_dir'],
            overwrite_cache = config['overwrite_cache'],
            equalize = config['equalize']
            )


    def train(self):
        for data_train, data_val in self.data_builder:
            input_normalizer = InputNormalizer(data_train)
            dataloader_train = DataLoader(data_train, 
                                        batch_size=self.config['batch_size'], 
                                        shuffle=True, 
                                        num_workers=self.config['batch_size'], 
                                        collate_fn=input_normalizer.collate_fn)
            dataloader_val = DataLoader(data_val, 
                                        batch_size=self.config['batch_size'], 
                                        shuffle=True, 
                                        num_workers=self.config['batch_size'], 
                                        collate_fn=input_normalizer.collate_fn)
            net = self.config['network'](self.data_builder.num_classes, self.add_softmax, self.config).to(self.device)
            self.run_training(net, dataloader_train)
            self.run_validation(net, dataloader_train)
            self.run_validation(net, dataloader_val)

        
    def run_training(self, model, dataloader):
        self.optimizer = optim.SGD(model.parameters(), lr=self.config['learning_rate'])
        model.train()
        for epoch in range(self.config['num_epochs']):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (batch, label, sequences_lengths) in enumerate(dataloader):
                batch = batch.to(self.device)
                label = label.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(batch, sequences_lengths)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            print(running_loss)

        print('Finished Training')
        
    def run_validation(self, model, dataloader, verbose=False):
        model.eval()
        accurate_predictions = 0
        num_records = 0
        with torch.no_grad():
            for i, (batch, label, sequences_lengths) in enumerate(dataloader):
                batch = batch.to(self.device)
                outputs = model(batch, sequences_lengths).cpu()
                predictions = np.argmax(outputs, axis=1)
                accurate_predictions += sum(predictions==label).item()
                num_records += len(predictions)
                if verbose:
                    print(outputs)
                    print(label)
        print(f"""Finished Evaluation: 
            {accurate_predictions}/{num_records} 
            ({round(accurate_predictions/num_records*100, 2)}%) accurate""")