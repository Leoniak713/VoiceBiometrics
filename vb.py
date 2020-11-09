import os
import time
from itertools import islice

from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchviz import make_dot, make_dot_from_trace

from models import BiggerCNNLSTM
from data import VoxDataset, InputNormalizer, DataBuilder
from hps import HPSParser
from monitors import ExperimentMonitor, ScoresMonitor

def run_experiment(source_config):
    hps_parser = HPSParser(source_config)
    experiments_logs = []
    for _ in range(source_config['num_hps_runs']):
        config, drawn_variants = hps_parser.draw_config()
        print(drawn_variants)
        trainer = Trainer(config)
        start_time = time.time()
        best_score = trainer.run_training()
        training_time = time.time() - start_time
        drawn_variants['Accuracy'] = best_score
        drawn_variants['Training time'] = training_time
        experiments_logs.append(drawn_variants)
    experiment_df = pd.DataFrame(experiments_logs)\
        .sort_values('Accuracy', ascending=False, inplace=False, ignore_index=True)
    return experiment_df

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
            preprocessing_params = config['preprocessing'], 
            num_folds = self.config['num_folds'],
            cache_dir = config['cache_dir'],
            overwrite_cache = config['overwrite_cache'],
            equalize = config['equalize']
            )
        self.num_classes = self.data_builder.num_classes


    def run_training(self):
        experiment_monitor = ExperimentMonitor()
        for fold in islice(self.data_builder, self.config['num_fold_trainings']):
            model = Model(
                fold, 
                self.num_classes, 
                self.add_softmax, 
                self.criterion, 
                self.config
            )
            model.train()
            experiment_monitor.add_scores_monitor(model.scores_monitor)
        
        best_score = experiment_monitor.show_scores()
        return best_score

    def visualize_model(self):
        model = Model(
            self.data_builder[0], 
            self.num_classes, 
            self.add_softmax, 
            self.criterion, 
            self.config
            )
        dummy_input = torch.randn(
            self.config['batch_size'], 
            100, 
            self.config['preprocessing']['n_mfcc']
            ).cuda()
        return make_dot(
            model.network(dummy_input), 
            params=dict(model.network.named_parameters())
        )


class Model:
    def __init__(self, fold, num_classes, add_softmax, criterion, config):
        self.config = config
        self.device = config['device']
        self.criterion = criterion
        data_train, data_val, input_normalizer = fold
        self.dataloader_train = DataLoader(
            data_train, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['batch_size'],
            drop_last=True,
            collate_fn=input_normalizer.collate_fn
            )
        self.dataloader_val = DataLoader(
            data_val, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['batch_size'],
            drop_last=True,
            collate_fn=input_normalizer.collate_fn
            )
        self.network = self.config['network'](
            num_classes, 
            add_softmax, 
            self.config
            ).to(self.device)
        self.scores_monitor = ScoresMonitor(data_train, data_val)


    def train(self):
        optimizer = optim.SGD(self.network.parameters(), lr=self.config['learning_rate'])
        scheduler = StepLR(
            optimizer, 
            **self.config['scheduler']
            )
        for _ in tqdm(range(self.config['num_epochs'])):
            self.network.train()
            for batch, label, _ in self.dataloader_train:
                optimizer.zero_grad()
                batch = batch.to(self.device)
                label = label.to(self.device)
                outputs = self.network(batch)
                loss = self.criterion(outputs, label)
                loss.backward()
                optimizer.step()
                self.scores_monitor.train_monitor.add_loss_and_predictions(loss, outputs, label)
            self.scores_monitor.train_monitor.append_epoch_loss_and_score()
            self.validate()
            scheduler.step()
        print('Finished Training')
        
    def validate(self, verbose=False):
        self.network.eval()
        with torch.no_grad():
            for batch, label, _ in self.dataloader_val:
                batch = batch.to(self.device)
                label = label.to(self.device)
                outputs = self.network(batch)
                loss = self.criterion(outputs, label)
                self.scores_monitor.val_monitor.add_loss_and_predictions(loss, outputs, label)
                if verbose:
                    print(outputs)
                    print(label)
            self.scores_monitor.val_monitor.append_epoch_loss_and_score()

