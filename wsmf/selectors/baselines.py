from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor

from .selector import WarmstartHpSelector
from .utils import get_ranks_of_hp_configurations


class RandomHpSelector(WarmstartHpSelector):

    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return list(
            np.random.choice(
                len(self.configurations), size=n_configurations, replace=False
            )
        )


class RankBasedHpSelector(WarmstartHpSelector):

    def __init__(
        self,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict[str, Any]],
    ):
        super().__init__(metadataset, landmarkers, configurations)
        self.ranks = get_ranks_of_hp_configurations(
            np.stack(
                [landmarker.cpu().numpy() for landmarker in self.landmarkers]
            )
        )

    def propose_configurations_idx(
        self, dataset: Tuple[Tensor, Tensor], n_configurations: int
    ) -> list[int]:
        return self.ranks[:n_configurations]


class LandmarkerHpSelector(WarmstartHpSelector):

    def __init__(
        self,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict[str, Any]],
    ):
        super().__init__(metadataset, landmarkers, configurations)

    def propose_configurations_idx(
        self, landmarkers: Tensor, n_configurations: int  # type: ignore
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
