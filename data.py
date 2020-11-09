import os
import shutil

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

class Fold:
    def __init__(self):
        self.metadata = []
        self.user_fold_count = 0
        
    def __len__(self):
        return len(self.metadata)
        
    def extend(self, data):
        self.metadata.extend(data)
        self.user_fold_count += len(data)
        
    def zero_user_counter(self):
        self.user_fold_count = 0

class FoldsBuilder:
    def __init__(self, num_folds, equalize=True):
        self.folds = [Fold() for _ in range(num_folds)]
        self.equalize = equalize
        
    def __iter__(self):
        return (fold for fold in self.folds)

    def __getitem__(self, idx):
        return self.folds[idx]
        
    def zero_counters(self):
        for fold in self.folds:
            fold.zero_user_counter()
            
    def _get_smallest_fold(self):
        smallest_fold = self.folds[0]
        for fold in self.folds:
            if fold.user_fold_count < smallest_fold.user_fold_count \
                or (
                    self.equalize \
                    and (fold.user_fold_count == smallest_fold.user_fold_count) \
                    and (len(fold) < len(smallest_fold))
                   ):
                smallest_fold = fold
        return smallest_fold
    
    def add(self, data):
        smallest_fold = self._get_smallest_fold()
        smallest_fold.extend(data)

class DataBuilder:
    def __init__(
        self, 
        data_path, 
        onehot_encoding, 
        preprocessing_params, 
        num_folds, 
        cache_dir, 
        overwrite_cache, 
        equalize
        ):
        self.onehot_encoding = onehot_encoding
        self.preprocessing_params = preprocessing_params
        cache_dir_name = '_'.join([f'{param}_{value}' for param, value in sorted(preprocessing_params.items())])
        self.cache_dir = os.path.join(cache_dir, cache_dir_name)
        self.equalize = equalize
        if os.path.exists(self.cache_dir) and overwrite_cache:
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok = True)
        self.folds, self.num_classes = self._get_folds(data_path, num_folds)

    def _get_folds(self, data_path, num_folds):
        fold_paths = [os.path.join(self.cache_dir, f'fold_{i}.csv') for i in range(num_folds)]

        if all([os.path.exists(fold_path) for fold_path in fold_paths]):
            metadata_folds = [pd.read_csv(fold_path) for fold_path in fold_paths]
            num_classes = pd.concat(metadata_folds)['label'].nunique()
            print('Folds loaded from cache')

        else:
            folds = FoldsBuilder(num_folds, self.equalize)
            users = os.listdir(data_path)
            num_classes = len(users)
            
            for user_id, user in enumerate(users):
                folds.zero_counters()
                user_metadata = []
                user_path = os.path.join(data_path, user)
                
                for user_dir in os.listdir(user_path):
                    user_dir_path = os.path.join(user_path, user_dir)
                    dir_files = os.listdir(user_dir_path)
                    metadata = [(os.path.join(user_dir_path, audio_file), user_id) for audio_file in dir_files]
                    user_metadata.append(metadata)
                    
                for metadata in sorted(user_metadata, key=lambda x: len(x), reverse=True):
                    folds.add(metadata)
                    
            metadata_folds = [pd.DataFrame(fold.metadata, columns=['filepath', 'label']) for fold in folds]

            for fold_path, metadata_fold in zip(fold_paths, metadata_folds):
                metadata_fold.to_csv(fold_path, index=False)

            print('Folds created')
        return metadata_folds, num_classes

    def __len__(self):
        return len(self.folds)
    
    def __getitem__(self, idx):
        train_metadata = pd.concat(
            [fold for i, fold in enumerate(self.folds) if i != idx],
            ignore_index=True
            )
        val_metadata = self.folds[idx].reset_index(drop=True)
        train_dataset = self._get_dataset(train_metadata, idx)
        val_dataset = self._get_dataset(val_metadata, idx)
        input_normalizer = InputNormalizer(train_dataset)
        return train_dataset, val_dataset, input_normalizer

    def _get_dataset(self, metadata, fold_id):
        return VoxDataset(
            metadata, 
            self.num_classes, 
            self.onehot_encoding, 
            self.preprocessing_params, 
            self.cache_dir,
            fold_id
            )


class VoxDataset(Dataset):
    def __init__(self, metadata, num_classes, onehot_encoding, preprocessing_params, cache_dir, fold_id):
        self.metadata = metadata
        self.num_classes = num_classes
        self.label_fun = self._onehot if onehot_encoding else lambda x: x
        self.preprocessing_params = preprocessing_params
        self.cache_dir = cache_dir
        self.fold_id = fold_id
    
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
            assert mfcc_tensor.shape[1] == self.preprocessing_params['n_mfcc'], \
                f"""MFCC cache shape mismatch: {mfcc_tensor.shape[1]} 
                while {self.preprocessing_params['n_mfcc']} was given in config"""
        else:
            audio = librosa.core.load(filepath)
            mfcc_tensor = torch.tensor(
                librosa.feature.mfcc(
                    audio[0], 
                    **self.preprocessing_params
                    ).transpose()
                )
            torch.save(mfcc_tensor, cache_path)
        return mfcc_tensor, label_tensor
    
class InputNormalizer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=48,
            shuffle=False,
            num_workers=12,
            drop_last=False,
            collate_fn=self._internal_collate_fn
            )
        self.cache_dir = self.dataset.cache_dir
        self.fold_id = self.dataset.fold_id
        self.mean, self.std = self._get_mean_std()
        
    def _get_mean_std(self):
        mean_cache_path = os.path.join(self.cache_dir, f'normalizer_mean_{self.fold_id}.pt')
        std_cache_path = os.path.join(self.cache_dir, f'normalizer_std_{self.fold_id}.pt')
        if os.path.exists(mean_cache_path) and os.path.exists(std_cache_path):
            mean = torch.load(mean_cache_path)
            std = torch.load(std_cache_path)
            assert mean.shape[0] == std.shape[0] == self.dataset.preprocessing_params['n_mfcc'], \
                f"""Mean and std cache shape mismatch: {mean.shape[0]} and {std.shape[0]}
                while {self.dataset.preprocessing_params['n_mfcc']} was given in config"""
            print('Mean and std loaded from cache')
        else:
            records = 0
            mean = torch.zeros(self.dataset.preprocessing_params['n_mfcc'])
            std = torch.zeros(self.dataset.preprocessing_params['n_mfcc'])

            for mfcc in self.dataloader:
                sample_records = len(mfcc)
                sample_mean = torch.mean(mfcc, axis=0)
                mean = (mean * records + sample_mean * sample_records) / (records + sample_records)
                sample_std = torch.std(mfcc, axis=0)
                std = (std * records + sample_std * sample_records) / (records + sample_records)
                records += sample_records
            torch.save(mean, mean_cache_path)
            torch.save(std, std_cache_path)
            print('Mean and std calculated')
            
        return mean, std

    @staticmethod
    def _internal_collate_fn(batch):
        return torch.cat([mfcc for mfcc, _ in batch], 0)

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
