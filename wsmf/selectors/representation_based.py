from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from dataset2vec.model import Dataset2Vec
from torch import Tensor

from experiments_engine.metamodels.networks.d2v_new_loss import (
    Dataset2VecForHpo,
)
from experiments_engine.portfolio_selection import (
    get_ranks_of_hp_configurations,
)

from .selector import WarmstartHpSelector


class Dataset2VecHpSelector(WarmstartHpSelector):

    @torch.no_grad()
    def __init__(
        self,
        encoder_path: Path,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict],
        algorithm: Literal["greedy", "asmfo"] = "greedy",
    ):
        super().__init__(metadataset, landmarkers, configurations, algorithm)
        try:
            self.encoder = Dataset2Vec.load_from_checkpoint(encoder_path)
        except:
            self.encoder = Dataset2VecForHpo.load_from_checkpoint(encoder_path)
        self.encodings = [
            self.encoder(*dataset_from_db) for dataset_from_db in self.datasets
        ]

    @torch.no_grad()
    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        dataset_encoding = self.encoder(*dataset)
        distances = np.array(
            [
                torch.norm(dataset_encoding - encoding_from_db).cpu().numpy()
                for encoding_from_db in self.encodings
            ]
        )
        closest_datasets_idx = np.argpartition(distances, n_configurations)[
            :n_configurations
        ].tolist()
        return [
            self.best_configurations_idx[idx] for idx in closest_datasets_idx
        ]

    @torch.no_grad()
    def propose_configuration_idx_asmfo(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        dataset_encoding = self.encoder(*dataset)
        distances = np.array(
            [
                torch.norm(dataset_encoding - encoding_from_db).cpu().numpy()
                for encoding_from_db in self.encodings
            ]
        )
        closest_datasets_idx = np.argpartition(distances, n_configurations)[
            :n_configurations
        ].tolist()
        closest_datasets = torch.stack(
            [self.landmarkers[idx] for idx in closest_datasets_idx]
        )
        ranks_of_configurations = get_ranks_of_hp_configurations(
            closest_datasets.cpu().numpy()
        )
        return np.argpartition(ranks_of_configurations, n_configurations)[
            :n_configurations
        ].tolist()


class Dataset2VecHpSelectorMixedDistances(WarmstartHpSelector):

    @torch.no_grad()
    def __init__(
        self,
        encoder_path: Path,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict],
        algorithm: Literal["greedy", "asmfo"] = "greedy",
        n_closest: int = 3,
        n_furthest: int = 2,
    ):
        super().__init__(metadataset, landmarkers, configurations, algorithm)
        self.encoder = Dataset2Vec.load_from_checkpoint(encoder_path)
        self.encodings = [
            self.encoder(*dataset_from_db) for dataset_from_db in self.datasets
        ]
        self.n_closest = n_closest
        self.n_furthest = n_furthest

    @torch.no_grad()
    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        dataset_encoding = self.encoder(*dataset)
        distances = np.array(
            [
                torch.norm(dataset_encoding - encoding_from_db).cpu().numpy()
                for encoding_from_db in self.encodings
            ]
        )
        closest_datasets_idx = np.argpartition(distances, n_configurations)[
            : self.n_closest
        ].tolist()
        furthest_datasets_idx = np.argpartition(-distances, n_configurations)[
            : self.n_furthest
        ].tolist()
        return [
            self.best_configurations_idx[idx]
            for idx in closest_datasets_idx + furthest_datasets_idx
        ]
