from __future__ import annotations

from typing import Any

from dataset2vec import Dataset2Vec
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wsmf.metamodels.train.reconstruction import (
    LandmarkerReconstructionTrainingInterface,
)


class Dataset2VecForLandmarkerReconstruction(
    LandmarkerReconstructionTrainingInterface
):
    def __init__(
        self,
        landmarker_size: int,
        config: Dataset2VecConfig = Dataset2VecConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__(optimizer_config, landmarkers_reconstruction_loss)
        self.landmarker_size = landmarker_size
        self.dataset2vec = Dataset2Vec(config, optimizer_config)
        self.landmarker_reconstructor = nn.Sequential(
            nn.Linear(config.output_size, config.output_size),
            nn.GELU(),
            nn.Linear(config.output_size, landmarker_size),
        )
        self.save_hyperparameters()

    def forward(self, X: Tensor, y: Tensor) -> Any:
        dataset_representation = self.dataset2vec(X, y)
        return self.landmarker_reconstructor(dataset_representation)

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[dict[str, Any]]]:
        optimizer = self.optimizer_config.optimizer_cls(  # type: ignore
            self.parameters(),
            lr=self.optimizer_config.learning_rate,  # type: ignore
            weight_decay=self.optimizer_config.weight_decay,  # type: ignore
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.1)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train_loss",
                "frequency": 1,
            }
        ]

    @classmethod
    def initialize_from_pretrained(
        cls,
        landmarker_size: int,
        config: Dataset2VecConfig,
        optimizer_config: OptimizerConfig,
        pretrained_model: Dataset2Vec,
    ) -> Dataset2VecForLandmarkerReconstruction:
        model = cls(landmarker_size, config, optimizer_config)
        model.dataset2vec = pretrained_model
        return model


def landmarkers_reconstruction_loss(
    true_landmarkers: Tensor, predicted_landmarkers: Tensor
) -> Tensor:
    labels = true_landmarkers.to(predicted_landmarkers.device)
    return ((predicted_landmarkers - labels) ** 2).mean(dim=1).mean()
