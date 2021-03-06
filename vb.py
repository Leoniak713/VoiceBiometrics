from itertools import islice
import time
import typing as t
import os

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

from graphviz import Digraph
from torchviz import make_dot, make_dot_from_trace

from models import BiggerCNNLSTM
from data import Fold, VoxDataset, InputNormalizer, DataBuilder
from hps import HPSParser
from monitors import ExperimentMonitor, ScoresMonitor

def run_experiment(source_config: t.Dict) -> pd.DataFrame:
    hps_parser = HPSParser(source_config)
    experiments_logs = []
    for _ in range(source_config['num_hps_runs']):
        config, drawn_variants = hps_parser.draw_config()
        print(drawn_variants)
        trainer = Trainer(config)
        start_time = time.time()
        best_score, best_score_index, best_score_train_acc, best_score_loss = trainer.run_training()
        training_time = round((time.time() - start_time) / 60, 1)
        drawn_variants['Time'] = training_time
        drawn_variants['Loss'] = best_score_loss
        drawn_variants['Best epoch'] = best_score_index
        drawn_variants['Accuracy (train)'] = best_score_train_acc
        drawn_variants['Accuracy'] = best_score
        experiments_logs.append(drawn_variants)
    experiment_df = pd.DataFrame(experiments_logs)\
        .sort_values('Accuracy', ascending=False, inplace=False, ignore_index=True)
    return experiment_df

class Trainer:
    def __init__(self, config: t.Dict) -> None:
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


    def run_training(self) -> t.Tuple[float]:
        experiment_monitor = ExperimentMonitor(self.config)
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
        best_score_stats = experiment_monitor.show_scores()
        return best_score_stats

    def visualize_model(self) -> Digraph:
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
    def __init__(
        self, 
        fold: Fold, 
        num_classes: int, 
        add_softmax: bool, 
        criterion: nn.Module, 
        config: t.Dict
    ):
        self.config = config
        self.device = config['device']
        self.criterion = criterion
        data_train, data_val, input_normalizer = fold
        self.dataloader_train = DataLoader(
            data_train, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['batch_size'],
            drop_last=self.config['drop_last'],
            collate_fn=input_normalizer.collate_fn
            )
        self.dataloader_val = DataLoader(
            data_val, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['batch_size'],
            drop_last=self.config['drop_last'],
            collate_fn=input_normalizer.collate_fn
            )
        self.network = self.config['network'](
            num_classes, 
            add_softmax, 
            self.config
            ).to(self.device)
        self.scores_monitor = ScoresMonitor(data_train, data_val)


    def train(self) -> None:
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
        
    def validate(self, verbose: bool = False) -> None:
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

