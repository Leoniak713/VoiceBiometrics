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

class DataBuilder:
    def __init__(self, data_path, onehot_encoding, n_mfcc, num_folds, cache_dir):
        self.metadata, self.num_classes = self._get_metadata(data_path)
        self.onehot_encoding = onehot_encoding
        self.n_mfcc = n_mfcc
        self.folds = self._get_folds(num_folds)
        self.cache_dir = cache_dir

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
        metadata_df = pd.DataFrame(metadata, columns=['filepath', 'label'])
        return metadata_df, num_classes

    def _get_folds(self, num_folds):
        metadata_folds = [
            pd.concat(i) \
            for i in zip(*[np.array_split(group, num_folds) \
            for _, group in self.metadata.groupby('label')])
            ]
        return metadata_folds

    def __len__(self):
        return len(self.folds)
    
    def __getitem__(self, idx):
        train_metadata = pd.concat(
            [fold for i, fold in enumerate(self.folds) if i != idx],
            ignore_index=True
            )
        val_metadata = self.folds[idx].reset_index(drop=True)
        return self._get_dataset(train_metadata), self._get_dataset(val_metadata)

    def _get_dataset(self, metadata):
        return VoxDataset(
            metadata, 
            self.num_classes, 
            self.onehot_encoding, 
            self.n_mfcc, 
            self.cache_dir
            )


class VoxDataset(Dataset):
    def __init__(self, metadata, num_classes, onehot_encoding, n_mfcc, cache_dir):
        self.metadata = metadata
        self.num_classes = num_classes
        self.label_fun = self._onehot if onehot_encoding else lambda x: x
        self.n_mfcc = n_mfcc
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok = True)
    
    def _onehot(self, idx):
        vector = np.zeros(self.num_classes)
        vector[idx] = 1
        return vector
    
    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        return (self[idx] for idx in range(len(self)))
    
    def __getitem__(self, idx):
        recording_metadata = self.metadata.loc[idx]
        label_tensor = torch.tensor(self.label_fun(recording_metadata['label']))
        filepath = recording_metadata['filepath']
        cache_name = '_'.join(filepath.split('/')[-2:])
        cache_path = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_path):
            mfcc_tensor = torch.load(cache_path)
            assert mfcc_tensor.shape[1] == self.n_mfcc, 'Cache shape mismatch'
        else:
            audio = librosa.core.load(filepath)
            mfcc_tensor = torch.tensor(librosa.feature.mfcc(audio[0], n_mfcc=self.n_mfcc).transpose())
            torch.save(mfcc_tensor, cache_path)
        return mfcc_tensor, label_tensor
    
class InputNormalizer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mean, self.std = self._calculate_mean_std()
        
    def _calculate_mean_std(self):
        records = 0
        mean = torch.zeros(self.dataset.n_mfcc)
        std = torch.zeros(self.dataset.n_mfcc)

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
