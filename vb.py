import os
import time
from itertools import islice

from tqdm import tqdm
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
from torch.optim.lr_scheduler import StepLR

from torchviz import make_dot, make_dot_from_trace

from models import BiggerCNNLSTM
from data import VoxDataset, InputNormalizer, DataBuilder
from hps import HPSParser

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
        fold_scores = []
        data_size = 0
        for fold in islice(self.data_builder, self.config['num_fold_trainings']):
            model = Model(
                fold, 
                self.num_classes, 
                self.add_softmax, 
                self.criterion, 
                self.config
            )
            model.train()
            model.scores_monitor.plot_scores()
            val_size = len(fold[1])
            fold_scores.append((model.scores_monitor.get_scores(), val_size))
            data_size += val_size
        
        best_score = max(sum([np.array(scores) * val_size / data_size \
            for scores, val_size in fold_scores]))
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
            print(scheduler.get_lr())
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

class ScoresMonitor:
    def __init__(self, data_train, data_val):
        self.data_train = data_train
        self.data_val = data_val
        self.baseline_accuracy = self.get_baseline_accuracy()
        self.num_classes = len(self.data_train.metadata['label'].unique())
        self.train_monitor = ScoreMonitor()
        self.val_monitor = ScoreMonitor(self.num_classes)

    def get_baseline_accuracy(self):
        mode_class = self.data_train.metadata['label'].value_counts().index[0]
        return len(self.data_val.metadata.query(f'label=={mode_class}')) \
            / len(self.data_val) * 100

    def plot_scores(self):
        fig, axs = plt.subplots(ncols=3, figsize=(30,7))
        axs[0].plot(self.train_monitor.epochs_loss, label='Train loss')
        axs[0].plot(self.val_monitor.epochs_loss, label='Valid loss')
        axs[0].legend(loc='best')

        baseline_accuracy = [self.baseline_accuracy] * len(self.train_monitor.epochs_loss)
        axs[1].plot(self.train_monitor.epochs_scores, label='Train acc')
        axs[1].plot(self.val_monitor.epochs_scores, label='Valid acc')
        axs[1].plot(baseline_accuracy, '.', label='Baseline acc')
        axs[1].legend(loc='best')

        ax = sns.heatmap(self.val_monitor.confusion_matrixes[-1], annot=True)
        ax.set(xlabel='Actual', ylabel='Predicted')
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelbottom = False, labeltop=True)
        axs[2] = ax
        plt.show()

    def get_scores(self):
        return self.val_monitor.epochs_scores


class ScoreMonitor:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.loss = 0
        self.epochs_loss = list()
        self.accurate_predictions = 0
        self.num_records = 0
        self.epochs_scores = list()
        self.confusion_matrixes = None
        if self.num_classes:
            self.confusion_matrixes = list()
            self.epoch_confusion_matrix = np.zeros([self.num_classes, self.num_classes])

    def add_loss_and_predictions(self, loss, outputs, labels):
        self.loss += loss.item()
        predictions = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        labels = labels.cpu().detach().numpy()
        self.accurate_predictions += sum(predictions==labels).item()
        self.num_records += len(predictions)
        if self.confusion_matrixes is not None:
            for prediction, label in zip(predictions, labels):
                self.epoch_confusion_matrix[prediction, label] += 1

    def append_epoch_loss_and_score(self):
        self.epochs_loss.append(self.loss / self.num_records)
        self.epochs_scores.append(round(self.accurate_predictions / self.num_records*100, 2))
        self.accurate_predictions = 0
        self.num_records = 0
        self.loss = 0
        if self.confusion_matrixes is not None:
            self.confusion_matrixes.append(self.epoch_confusion_matrix)
            self.epoch_confusion_matrix = np.zeros([self.num_classes, self.num_classes])

