from __future__ import annotations

from copy import deepcopy
from typing import Any, Tuple

import numpy as np
from torch import Tensor

from wsmf.metamodels.data.dataset import EncoderHpoDataset


class LandmarkerReconstructionLoader:
    """
    Loader for generating batches of data for landmarker reconstruction.

    Creates batches of data from an `EncoderHpoDataset` for landmarker
    reconstruction tasks. Each batch is triplet
    (features, targets, landmarkers)

    Parameters
    ----------
    dataset : EncoderHpoDataset
        Dataset containing input, target, and landmark data.
    batch_size : int
        Size of each batch.
    shuffle : bool, optional
        Whether to shuffle the dataset before creating batches.
        Default is False.

    Attributes
    ----------
    dataset : EncoderHpoDataset
        Stored dataset.
    batch_size : int
        Size of each batch.
    shuffle : bool
        Whether to shuffle the dataset.
    n_datasets : int
        Number of datasets in the dataset.
    dataset_names : list
        List of dataset names.

    Methods
    -------
    __next__()
        Generates the next batch of data.
    __iter__()
        Returns an iterator over the loader.
    __len__()
        Returns the number of batches.
    """

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
    ) -> list[
        Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, dict[str, Any]]
    ]:
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

    def __iter__(self) -> LandmarkerReconstructionLoader:
        if self.shuffle:
            self.sample_indices = np.random.permutation(self.n_datasets)
        return deepcopy(self)

    def __len__(self) -> int:
        return self.n_datasets // self.batch_size + 1

    def __generate_sample(
        self, dataset_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dataset_name = self.dataset_names[dataset_idx]

        return self.dataset[dataset_name]
