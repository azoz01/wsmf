from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from torch import Tensor

from engine.metamodels.networks.d2v_reconstruction import (
    Dataset2VecForLandmarkerReconstruction,
)
from engine.portfolio_selection import get_ranks_of_hp_configurations

from .selector import WarmstartHpSelector


class Dataset2VecReconstructionHpSelector(WarmstartHpSelector):

    def __init__(
        self,
        encoder_path: Path,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict],
        algorithm: Literal["greedy", "asmfo"] = "greedy",
    ):
        super().__init__(metadataset, landmarkers, configurations, algorithm)
        self.encoder = (
            Dataset2VecForLandmarkerReconstruction.load_from_checkpoint(
                encoder_path
            )
        )

    @torch.no_grad()
    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        predicted_landmarkers = self.encoder(*dataset)
        distances = np.array(
            [
                torch.norm(predicted_landmarkers - landmarker_from_db)
                .cpu()
                .numpy()
                for landmarker_from_db in self.landmarkers
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
        predicted_landmarkers = self.encoder(*dataset)
        distances = np.array(
            [
                torch.norm(predicted_landmarkers - landmarker_from_db)
                .cpu()
                .numpy()
                for landmarker_from_db in self.landmarkers
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


class Dataset2VecReconstructionHpSelectorMixedDistances(WarmstartHpSelector):

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
        self.encoder = (
            Dataset2VecForLandmarkerReconstruction.load_from_checkpoint(
                encoder_path
            )
        )
        self.n_closest = n_closest
        self.n_furthest = n_furthest

    @torch.no_grad()
    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        predicted_landmarkers = self.encoder(*dataset)
        distances = np.array(
            [
                torch.norm(predicted_landmarkers - encoding_from_db)
                .cpu()
                .numpy()
                for encoding_from_db in self.landmarkers
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

    def propose_configuration_idx_asmfo(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        raise NotImplementedError()
