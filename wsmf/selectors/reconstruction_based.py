from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor

from wsmf.metamodels.train import LandmarkerReconstructionTrainingInterface

from .selector import WarmstartHpSelector


class ReconstructionBasedHpSelector(WarmstartHpSelector):

    def __init__(
        self,
        encoder: LandmarkerReconstructionTrainingInterface,
        metadataset: dict[str, Tuple[Tensor, Tensor]],
        landmarkers: dict[str, Tensor],
        configurations: list[dict[str, Any]],
    ):
        super().__init__(metadataset, landmarkers, configurations)
        self.encoder = encoder

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
