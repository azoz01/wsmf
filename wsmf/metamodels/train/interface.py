from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

import pytorch_lightning as pl
import torch
from dataset2vec.config import OptimizerConfig
from torch import Tensor, stack
from torch.optim.optimizer import Optimizer


class TrainingInterface(pl.LightningModule, ABC):

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        loss_function: Callable[[Tensor, Tensor], Tensor],
    ):
        super().__init__()
        self.loss_function = loss_function
        self.optimizer_config = optimizer_config
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:  # type: ignore
        pass

    @abstractmethod
    def calculate_datasets_similarity(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        pass

    @abstractmethod
    def configure_optimizers(  # type: ignore
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        pass

    # Training phase
    def on_train_epoch_start(self) -> None:
        self.training_labels: list[Tensor] = []
        self.training_predictions: list[Tensor] = []

    def training_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
    ) -> Mapping[str, Tensor]:
        labels, similarities = self.extract_labels_and_similarities_from_batch(
            batch
        )
        loss = self.calculate_loss(labels, similarities)
        self.log("train_step_loss", loss, prog_bar=True, batch_size=len(batch))
        return {"loss": loss, "predictions": similarities}

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if isinstance(outputs, Mapping):
            self.training_predictions.append(outputs["predictions"])
        else:
            raise TypeError("outptus should have type Mapping[str, Any]")
        if (
            not isinstance(batch[0][-1], Tensor)
            or len(batch[0][-1].shape) == 0
        ):
            self.training_labels.append(Tensor([obs[-1] for obs in batch]))
        else:
            self.training_labels.append(
                torch.stack([obs[-1] for obs in batch])
            )

    def on_train_epoch_end(self) -> None:
        training_predictions = torch.concat(self.training_predictions, dim=0)
        training_labels = torch.concat(self.training_labels, dim=0).to(
            training_predictions.device
        )
        self.log(
            "train_loss",
            self.calculate_loss(training_labels, training_predictions),
        )

    # Validation phase
    def on_validation_epoch_start(self) -> None:
        self.validation_labels: list[Tensor] = []
        self.validation_predictions: list[Tensor] = []

    def validation_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
    ) -> Mapping[str, Tensor]:
        labels, similarities = self.extract_labels_and_similarities_from_batch(
            batch
        )
        loss = self.calculate_loss(labels, similarities)

        return {"loss": loss, "predictions": similarities}

    def on_validation_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if isinstance(outputs, Mapping):
            self.validation_predictions.append(outputs["predictions"])
        else:
            raise TypeError("outptus should have type Mapping[str, Any]")
        if (
            not isinstance(batch[0][-1], Tensor)
            or len(batch[0][-1].shape) == 0
        ):
            self.validation_labels.append(Tensor([obs[-1] for obs in batch]))
        else:
            self.validation_labels.append(
                torch.stack([obs[-1] for obs in batch])
            )

    def on_validation_epoch_end(self) -> None:
        validation_predictions = torch.concat(
            self.validation_predictions, dim=0
        )
        validation_labels = torch.concat(self.validation_labels, dim=0).to(
            validation_predictions.device
        )
        self.log(
            "val_loss",
            self.calculate_loss(validation_labels, validation_predictions),
        )

    # Utility functions
    def calculate_loss(self, labels: Tensor, predictions: Tensor) -> Tensor:
        labels = labels.to(predictions.device)
        return self.loss_function(labels, predictions)

    def extract_labels_and_similarities_from_batch(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
    ) -> tuple[Tensor, Tensor]:
        similarities = []
        labels = []
        for X1, y1, X2, y2, label in batch:
            similarities.append(
                self.calculate_datasets_similarity(X1, y1, X2, y2)
            )
            labels.append(label)
        return Tensor(labels), stack(similarities).view(-1)
