from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from dataset2vec import Dataset2Vec
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from wsmf.metamodels.train.reconstruction import (
    LandmarkerReconstructionTrainingInterface,
)


class Dataset2VecForLandmarkerReconstruction(
    LandmarkerReconstructionTrainingInterface
):
    """
    Dataset2Vec-based model for landmarkers reconstruction.
    Now it only supports landmarkers based on ROC AUC

    Parameters
    ----------
    landmarker_size : int
        Size of the landmarkers.
    config : Dataset2VecConfig, optional
        Configuration for the Dataset2Vec model.
        Defaults to Dataset2VecConfig().
    optimizer_config : OptimizerConfig, optional
        Configuration for the optimizer.
        Defaults to OptimizerConfig().

    Attributes
    ----------
    landmarker_size : int
        Size of the landmarkers.
    dataset2vec : Dataset2Vec
        The Dataset2Vec model instance.
    landmarker_reconstructor : nn.Sequential
        Neural network module for landmarkers reconstruction.

    Methods
    -------
    forward(X, y)
        Forward pass of the model.
    configure_optimizers()
        Configures the optimizer and learning rate scheduler.
    """

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
        return roc_auc_activation(
            self.landmarker_reconstructor(dataset_representation)
        )

    def configure_optimizers(  # type: ignore
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = self.optimizer_config.optimizer_cls(
            self.parameters(),
            lr=self.optimizer_config.learning_rate,
            weight_decay=self.optimizer_config.weight_decay,
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


def landmarkers_reconstruction_loss(
    true_landmarkers: Tensor, predicted_landmarkers: Tensor
) -> Tensor:
    """
    Loss function for landmarker reconstruction.

    Calculates the mean squared error between true and predicted landmarkers.

    Parameters
    ----------
    true_landmarkers : Tensor
        Tensor of true landmarker values.
    predicted_landmarkers : Tensor
        Tensor of predicted landmarker values.

    Returns
    -------
    Tensor
        Mean squared error loss between true and predicted landmarkers.
    """
    labels = true_landmarkers.to(predicted_landmarkers.device)
    return ((predicted_landmarkers - labels) ** 2).mean(dim=1).mean()  # type: ignore # noqa: E501


def roc_auc_activation(x: Tensor) -> Tensor:
    return (1 - F.relu(1 - F.relu(x))) / 2 + 0.5  # type: ignore
