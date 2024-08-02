from abc import ABC, abstractmethod
from typing import Literal, Tuple

import numpy as np
from torch import Tensor


class WarmstartHpSelector(ABC):

    def __init__(
        self,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict],
        algorithm: Literal["greedy", "asmfo"] = "greedy",
    ):
        self.landmarkers_orig = landmarkers
        self.configurations = configurations
        self.metadataset = metadataset
        self.algorithm = algorithm

        self.datasets_names = list(sorted(metadataset.keys()))
        self.best_configurations_idx = [
            np.argmax(landmarkers[dataset_name].cpu().numpy())
            for dataset_name in self.datasets_names
        ]
        self.datasets = [
            metadataset[dataset_name] for dataset_name in self.datasets_names
        ]
        self.landmarkers = [
            landmarkers[dataset_name] for dataset_name in self.datasets_names
        ]

    def propose_configurations(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[dict]:
        if self.algorithm == "greedy":
            idx = self.propose_configurations_idx(dataset, n_configurations)
        elif self.algorithm == "asmfo":
            idx = self.propose_configuration_idx_asmfo(
                dataset, n_configurations
            )
        return [self.configurations[i] for i in idx]

    @abstractmethod
    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        pass

    @abstractmethod
    def propose_configuration_idx_asmfo(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        pass
