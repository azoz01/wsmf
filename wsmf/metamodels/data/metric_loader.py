from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np
from torch import Tensor

from .datasets import EncoderHpoDataset


class EncoderMetricLearningLoader:

    def __init__(
        self, dataset: EncoderHpoDataset, n_batches: int, batch_size: int
    ):
        self.dataset = dataset
        self.n_batches = n_batches
        self.batch_size = batch_size

        self.n_datasets = len(dataset)
        self.dataset_names = dataset.dataset_names
        self.counter = 0

    def __next__(self) -> list[Tuple[Tensor, Tensor, Tensor, Tensor, float]]:
        if self.counter == self.n_batches:
            raise StopIteration()
        self.counter += 1
        return [self.__generate_sample() for _ in range(self.batch_size)]

    def __iter__(self) -> EncoderMetricLearningLoader:
        return deepcopy(self)

    def __len__(self):
        return self.n_batches

    def __generate_sample(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
        dataset1_idx, dataset2_idx = np.random.choice(
            self.n_datasets, size=2, replace=False
        )
        dataset1_name, dataset2_name = (
            self.dataset_names[dataset1_idx],
            self.dataset_names[dataset2_idx],
        )
        dataset1_X, dataset1_y, landmarkers1 = self.dataset[dataset1_name]
        dataset2_X, dataset2_y, landmarkers2 = self.dataset[dataset2_name]
        return (
            dataset1_X,
            dataset1_y,
            dataset2_X,
            dataset2_y,
            self.__calculcate_landmarkers_similarity(
                landmarkers1, landmarkers2
            ),
        )

    def __calculcate_landmarkers_similarity(
        self, landmarkers1: Tensor, landmarkers2: Tensor
    ) -> float:
        return ((landmarkers1 - landmarkers2) ** 2).mean().item()
