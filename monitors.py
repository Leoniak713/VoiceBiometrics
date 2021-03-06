import typing as t

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from data import VoxDataset


class ScoreMonitor:
    def __init__(self, num_classes: t.Union[int, None] = None) -> None:
        self.num_classes = num_classes
        self.loss = 0
        self.epochs_loss = list()
        self.accurate_predictions = 0
        self.sum_records = 0
        self.num_records = 0
        self.epochs_accurate_predictions = list()
        self.confusion_matrixes = None
        if self.num_classes:
            self.confusion_matrixes = list()
            self.epoch_confusion_matrix = np.zeros(
                [self.num_classes, self.num_classes], 
                dtype=int
                )

    def __getitem__(self, key: str) -> t.Any:
        return getattr(self, key)

    def add_loss_and_predictions(
        self, 
        loss: torch.Tensor, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> None:
        self.loss += loss.item()
        predictions = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        labels = labels.cpu().detach().numpy()
        self.accurate_predictions += sum(predictions==labels).item()
        self.num_records += len(predictions)
        if self.confusion_matrixes is not None:
            for prediction, label in zip(predictions, labels):
                self.epoch_confusion_matrix[prediction, label] += 1

    def append_epoch_loss_and_score(self) -> None:
        self.sum_records = self.num_records
        self.epochs_loss.append(self.loss)
        self.epochs_accurate_predictions.append(self.accurate_predictions)
        self.accurate_predictions = 0
        self.num_records = 0
        self.loss = 0
        if self.confusion_matrixes is not None:
            self.confusion_matrixes.append(self.epoch_confusion_matrix)
            self.epoch_confusion_matrix = np.zeros([self.num_classes, self.num_classes])


class ScoresMonitor:
    def __init__(self, data_train: VoxDataset, data_val: VoxDataset) -> None:
        self.data_train = data_train
        self.data_val = data_val
        self.num_classes = len(self.data_train.metadata['label'].unique())
        self.train_monitor = ScoreMonitor()
        self.val_monitor = ScoreMonitor(self.num_classes)

    def __getitem__(self, key: str) -> t.Any:
        return getattr(self, key)

    def get_train_size(self) -> int:
        return len(self.data_train)

    def get_val_size(self) -> int:
        return len(self.data_val)

    def get_baseline_accurate_predictions(self) -> int:
        mode_class = self.data_train.metadata['label'].value_counts().index[0]
        return len(self.data_val.metadata.query(f'label=={mode_class}'))


class ExperimentMonitor:
    def __init__(self, config: t.Dict) -> None:
        self.config = config
        self.scores_monitors = []

    def add_scores_monitor(self, scores_monitor: ScoresMonitor) -> None:
        self.scores_monitors.append(scores_monitor)

    def _get_monitors_stats_sum(
        self, 
        monitor_type: str, 
        stat: str, 
        iterable_stat: bool
    ) -> float:
        transform_func = np.array if iterable_stat else lambda x: x
        return sum(transform_func(scores_monitor[monitor_type][stat]) \
            for scores_monitor in self.scores_monitors)

    def _get_baseline_accurate_predictions(self) -> float:
        return sum(scores_monitor.get_baseline_accurate_predictions() \
            for scores_monitor in self.scores_monitors)

    def show_scores(self) -> t.Tuple[float]:
        train_records = self._get_monitors_stats_sum('train_monitor', 'sum_records', False)
        val_records = self._get_monitors_stats_sum('val_monitor', 'sum_records', False)
        train_accuracy = self._get_monitors_stats_sum(
            'train_monitor', 
            'epochs_accurate_predictions',
            True
        ) / train_records * 100
        val_accuracy = self._get_monitors_stats_sum(
            'val_monitor', 
            'epochs_accurate_predictions', 
            True
        ) / val_records * 100
        train_loss = self._get_monitors_stats_sum(
            'train_monitor', 
            'epochs_loss', 
            True
        ) / train_records
        val_loss = self._get_monitors_stats_sum('val_monitor', 'epochs_loss', True) / val_records
        val_confusion_matrixes = self._get_monitors_stats_sum(
            'val_monitor', 
            'confusion_matrixes', 
            True
        )
        baseline_accuracy = self._get_baseline_accurate_predictions() / val_records * 100

        best_score_index, best_score = max(enumerate(val_accuracy), key=lambda p: p[1])
        best_score_train_acc = round(train_accuracy[best_score_index], 2)
        best_score_loss = round(val_loss[best_score_index], 4)
        print(round(best_score, 2))

        _, axs = plt.subplots(ncols=2, figsize=(30,7))
        axs[0].plot(train_loss, label='Train loss')
        axs[0].plot(val_loss, label='Valid loss')
        axs[0].legend(loc='best')

        axs[1].plot(train_accuracy, label='Train acc')
        axs[1].plot(val_accuracy, label='Valid acc')
        axs[1].plot([baseline_accuracy] * len(train_loss), '.', label='Baseline acc')
        axs[1].legend(loc='best')
        plt.show()

        plt.figure(figsize = (30, 18))
        ax = sns.heatmap(
            val_confusion_matrixes[best_score_index].astype(int), 
            annot=self.config['annotate_cm'], 
            fmt='d'
            )
        ax.set(xlabel='Actual', ylabel='Predicted')
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelbottom = False, labeltop=True)
        plt.show()
        return round(best_score, 2), best_score_index + 1, best_score_train_acc, best_score_loss
