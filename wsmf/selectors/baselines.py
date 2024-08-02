from typing import Literal, Tuple

import numpy as np
import torch
from torch import Tensor

from engine.portfolio_selection import get_ranks_of_hp_configurations

from .selector import WarmstartHpSelector


class RandomHpSelector(WarmstartHpSelector):

    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return np.random.choice(
            len(self.configurations), size=n_configurations, replace=False
        ).tolist()

    def propose_configuration_idx_asmfo(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return self.propose_configurations_idx(dataset, n_configurations)


class RankBasedHpSelector(WarmstartHpSelector):

    def __init__(
        self,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict],
        algorithm: Literal["greedy", "asmfo"],
    ):
        super().__init__(metadataset, landmarkers, configurations, algorithm)
        self.ranks = get_ranks_of_hp_configurations(
            np.stack(
                [landmarker.cpu().numpy() for landmarker in self.landmarkers]
            )
        )

    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return self.ranks[:n_configurations]

    def propose_configuration_idx_asmfo(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return self.propose_configurations_idx(dataset, n_configurations)


class LandmarkerHpSelector(WarmstartHpSelector):

    def __init__(
        self,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict],
        algorithm: Literal["greedy", "asmfo"] = "greedy",
    ):
        super().__init__(metadataset, landmarkers, configurations, algorithm)

    def propose_configurations_idx(
        self, landmarkers: Tensor, n_configurations: int
    ) -> list[int]:
        distances = np.array(
            [
                torch.norm(landmarkers - landmarkers_from_db).cpu().numpy()
                for landmarkers_from_db in self.landmarkers
            ]
        )
        closest_landmarkers_idx = np.argpartition(distances, n_configurations)[
            :n_configurations
        ].tolist()
        return [
            self.best_configurations_idx[idx]
            for idx in closest_landmarkers_idx
        ]

    def propose_configuration_idx_asmfo(
        self, landmarkers: Tensor, n_configurations: int
    ) -> list[int]:
        distances = np.array(
            [
                torch.norm(landmarkers - landmarkers_from_db).cpu().numpy()
                for landmarkers_from_db in self.landmarkers
            ]
        )
        closest_landmarkers_idx = np.argpartition(distances, n_configurations)[
            :n_configurations
        ].tolist()
        closest_landmarkers = torch.stack(
            [self.landmarkers[idx] for idx in closest_landmarkers_idx]
        )
        ranks_of_configurations = get_ranks_of_hp_configurations(
            closest_landmarkers.cpu().numpy()
        )
        return np.argpartition(ranks_of_configurations, n_configurations)[
            :n_configurations
        ].tolist()
