from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from dataset2vec import Dataset2Vec
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from torch import Tensor

from wsmf.metamodels.train.metric import MetricLearningTrainingInterface


class Dataset2VecForHpo(MetricLearningTrainingInterface):
    def __init__(
        self,
        config: Dataset2VecConfig = Dataset2VecConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__(optimizer_config, F.mse_loss)
        self.dataset2vec = Dataset2Vec(config, optimizer_config)

    @classmethod
    def initialize_from_pretrained(
        cls,
        config: Dataset2VecConfig,
        optimizer_config: OptimizerConfig,
        pretrained_model: Dataset2Vec,
    ) -> Dataset2VecForHpo:
        model = cls(config, optimizer_config)
        model.dataset2vec = pretrained_model
        return model

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        return self.dataset2vec(X, y)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        optimizer = self.optimizer_config.optimizer_cls(  # type: ignore
            self.parameters(),
            lr=self.optimizer_config.learning_rate,  # type: ignore
            weight_decay=self.optimizer_config.weight_decay,  # type: ignore
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
