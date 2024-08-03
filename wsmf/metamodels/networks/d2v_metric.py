from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from dataset2vec import Dataset2Vec
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from torch import Tensor
from torch.optim.optimizer import Optimizer

from wsmf.metamodels.train.metric import MetricLearningTrainingInterface


class Dataset2VecMetricLearning(MetricLearningTrainingInterface):
    """
    Dataset2Vec-based metric learning training interface.

    Implements metric learning training using the Dataset2Vec model.

    Parameters
    ----------
    config : Dataset2VecConfig, optional
        Configuration for the Dataset2Vec model.
        Defaults to Dataset2VecConfig().
    optimizer_config : OptimizerConfig, optional
        Configuration for the optimizer.
        Defaults to OptimizerConfig().

    Attributes
    ----------
    dataset2vec : Dataset2Vec
        The Dataset2Vec model instance.

    Methods
    -------
    forward(X, y)
        Forward pass of the model.
    """

    def __init__(
        self,
        config: Dataset2VecConfig = Dataset2VecConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__(optimizer_config, F.mse_loss)
        self.dataset2vec = Dataset2Vec(config, optimizer_config)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        return self.dataset2vec(X, y)  # type: ignore

    def configure_optimizers(  # type: ignore
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = self.optimizer_config.optimizer_cls(
            self.parameters(),
            lr=self.optimizer_config.learning_rate,
            weight_decay=self.optimizer_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            }
        ]
