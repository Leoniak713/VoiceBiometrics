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

class VoxDataset(Dataset):
    def __init__(self, data_path, onehot_encoding, n_mfcc):
        self.n_mfcc = n_mfcc
        self.metadata, self.num_classes = self._get_metadata(data_path)
        self.label_fun = self._onehot if onehot_encoding else lambda x: x
        
    @staticmethod
    def _get_metadata(data_path):
        metadata = []
        users = os.listdir(data_path)
        num_classes = len(users)
        for i, user in enumerate(users):
            user_path = os.path.join(data_path, user)
            for user_dir in os.listdir(user_path):
                user_dir_path = os.path.join(user_path, user_dir)
                for audio_file in os.listdir(user_dir_path):
                    metadata.append((os.path.join(user_dir_path, audio_file), i))
        return metadata, num_classes
    
    def _onehot(self, idx):
        vector = np.zeros(self.num_classes)
        vector[idx] = 1
        return vector
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        filename, label = self.metadata[idx]
        audio = librosa.core.load(filename)
        mfcc_tensor = torch.tensor(librosa.feature.mfcc(audio[0], n_mfcc=self.n_mfcc).transpose())
        label_tensor = torch.tensor(self.label_fun(label))
        return mfcc_tensor, label_tensor
    
#     @staticmethod
#     def collate_fn(batch):
#         x_sequences = list()
#         labels = list()
#         sequences_lengths = list()
#         for data, label in batch:
#             x_sequences.append(data)
#             labels.append(label)
#             sequences_lengths.append(len(data))
#         padded_sequences = pad_sequence(x_sequences, batch_first=True, padding_value=-300.)
#         labels_tensor = torch.stack(labels)
#         return padded_sequences, labels_tensor, sequences_lengths
    
class InputNormalizer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mean, self.std = self._calculate_mean_std()
        
    def _calculate_mean_std(self):
        records = 0
        mean = torch.zeros(10)
        std = torch.zeros(10)

        for mfcc, _ in self.dataset:
            sample_records = len(mfcc)
            sample_mean = torch.mean(mfcc, axis=0)
            mean = (mean * records + sample_mean * sample_records) / (records + sample_records)
            sample_std = torch.std(mfcc, axis=0)
            std = (std * records + sample_std * sample_records) / (records + sample_records)
            records += sample_records
            
        return mean, std

    def collate_fn(self, batch):
        max_length = max((len(sequence) for sequence, _ in batch))
        n_channels = max((sequence.shape[1] for sequence, _ in batch))
        padded_sequences = torch.zeros(len(batch), max_length, n_channels)
        
        labels = list()
        sequences_lengths = list()
        
        for i, (mfcc, label) in enumerate(batch):
            seq_len = len(mfcc)
            normalized_mfcc = (mfcc - self.mean) / self.std
            padded_sequences[i][max_length - seq_len : ] = normalized_mfcc
            labels.append(label)
            sequences_lengths.append(seq_len)
            
        labels_tensor = torch.stack(labels)
        return padded_sequences, labels_tensor, sequences_lengths
