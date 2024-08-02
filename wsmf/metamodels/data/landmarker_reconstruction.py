from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np
from torch import Tensor

from .datasets import EncoderHpoDataset


class GenericD2vHpoDataLoaderForHpo:

    def __init__(
        self,
        dataset: EncoderHpoDataset,
        batch_size: int,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_datasets = len(dataset)
        self.dataset_names = dataset.dataset_names
        self.sample_indices = (
            np.random.permutation(self.n_datasets)
            if self.shuffle
            else np.arange(self.n_datasets)
        )
        self.batch_counter = 0

    def __next__(
        self,
    ) -> list[Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, dict]]:
        start_index = self.batch_counter * self.batch_size
        end_index = (
            start_index + self.batch_size
            if start_index + self.batch_size <= self.n_datasets
            else self.n_datasets - 1
        )
        if start_index >= self.n_datasets:
            raise StopIteration()
        self.batch_counter += 1
        return [
            self.__generate_sample(self.sample_indices[idx])
            for idx in range(start_index, end_index)
        ]

    def __iter__(self) -> GenericD2vHpoDataLoaderForHpo:
        if self.shuffle:
            self.sample_indices = np.random.permutation(self.n_datasets)
        return deepcopy(self)

    def __len__(self):
        return self.n_datasets // self.batch_size + 1

    def __generate_sample(self, dataset_idx) -> Tuple[Tensor, Tensor, Tensor]:
        dataset_name = self.dataset_names[dataset_idx]

        return self.dataset[dataset_name]
