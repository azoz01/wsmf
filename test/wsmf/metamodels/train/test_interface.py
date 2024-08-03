from typing import Any
from unittest.mock import Mock, patch

from dataset2vec.config import OptimizerConfig
from torch import Tensor, rand
from torch.optim.optimizer import Optimizer

from wsmf.metamodels.train.interface import TrainingInterface


class MockImplementation(TrainingInterface):
    def forward(self, *args, **kwargs) -> Tensor:  # type: ignore
        return Tensor([0])

    def calculate_datasets_similarity(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        return Tensor([0.5])

    def configure_optimizers(  # type: ignore
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        return None  # type: ignore


def test_on_train_epoch_start() -> None:
    # Given
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))

    # When
    interface.on_train_epoch_start()

    # Then
    assert interface.training_labels == []
    assert interface.training_predictions == []


@patch(
    "wsmf.metamodels.train.interface"
    ".TrainingInterface.extract_labels_and_similarities_from_batch"
)
@patch("wsmf.metamodels.train.interface.TrainingInterface.calculate_loss")
def test_training_step(
    calculate_loss_mock: Mock,
    extract_labels_and_similarities_from_batch_mock: Mock,
) -> None:
    # Given
    extract_labels_and_similarities_from_batch_mock.return_value = (
        Tensor([1.0, 2.0, 3.0]),
        Tensor([4.0, 5.0, 6.0]),
    )
    calculate_loss_mock.return_value = Tensor([2.0])
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))

    # When
    actual_output = interface.training_step(
        [(Tensor([0]), Tensor([0]), Tensor([0]), Tensor([0]), 0)]
    )

    # Then
    calculate_loss_call_args = calculate_loss_mock.call_args
    assert (calculate_loss_call_args[0][0] == Tensor([1.0, 2.0, 3.0])).all()
    assert (calculate_loss_call_args[0][1] == Tensor([4.0, 5.0, 6.0])).all()
    assert list(actual_output.keys()) == ["loss", "predictions"]
    assert (actual_output["loss"] == Tensor([2.0])).all()
    assert (actual_output["predictions"] == Tensor([4.0, 5.0, 6.0])).all()


def test_on_train_batch_end() -> None:
    # Given
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))
    interface.on_train_epoch_start()

    # When
    interface.on_train_batch_end(
        {"loss": Tensor([2.0]), "predictions": Tensor([4.0, 5.0])},
        [
            (rand(10, 5), rand(10, 1), rand(5, 3), rand(5, 1), 1),
            (rand(10, 5), rand(10, 1), rand(5, 3), rand(5, 1), 0),
        ],
        0,
    )

    # Then
    assert len(interface.training_predictions) == 1
    assert (interface.training_predictions[0] == Tensor([4.0, 5.0])).all()
    assert len(interface.training_labels) == 1
    assert (interface.training_labels[0] == Tensor([1, 0])).all()


@patch("wsmf.metamodels.train.interface.TrainingInterface.calculate_loss")
def test_on_train_epoch_end(calculate_loss_mock: Mock) -> None:
    # Given
    calculate_loss_mock.return_value = Tensor([0.0])
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))
    interface.training_labels = [Tensor([1, 0]), Tensor([1, 1])]
    interface.training_predictions = [Tensor([0.1, 0.2]), Tensor([0.3, 0.4])]

    # When
    interface.on_train_epoch_end()

    # Then
    calculate_loss_mock.assert_called()
    calculate_loss_args = calculate_loss_mock.call_args
    assert (calculate_loss_args[0][0] == Tensor([1, 0, 1, 1])).all()
    assert (calculate_loss_args[0][1] == Tensor([0.1, 0.2, 0.3, 0.4])).all()


def test_on_validation_epoch_start() -> None:
    # Given
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))

    # When
    interface.on_validation_epoch_start()

    # Then
    assert interface.validation_labels == []
    assert interface.validation_predictions == []


@patch(
    "wsmf.metamodels.train.interface"
    ".TrainingInterface.extract_labels_and_similarities_from_batch"
)
@patch("wsmf.metamodels.train.interface.TrainingInterface.calculate_loss")
def test_validation_step(
    calculate_loss_mock: Mock,
    extract_labels_and_similarities_from_batch_mock: Mock,
) -> None:
    # Given
    extract_labels_and_similarities_from_batch_mock.return_value = (
        Tensor([1.0, 2.0, 3.0]),
        Tensor([4.0, 5.0, 6.0]),
    )
    calculate_loss_mock.return_value = Tensor([2.0])
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))

    # When
    actual_output = interface.validation_step(
        [(Tensor([0]), Tensor([0]), Tensor([0]), Tensor([0]), 0)]
    )

    # Then
    calculate_loss_call_args = calculate_loss_mock.call_args
    assert (calculate_loss_call_args[0][0] == Tensor([1.0, 2.0, 3.0])).all()
    assert (calculate_loss_call_args[0][1] == Tensor([4.0, 5.0, 6.0])).all()
    assert list(actual_output.keys()) == ["loss", "predictions"]
    assert (actual_output["loss"] == Tensor([2.0])).all()
    assert (actual_output["predictions"] == Tensor([4.0, 5.0, 6.0])).all()


def test_on_validation_batch_end() -> None:
    # Given
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))
    interface.on_validation_epoch_start()

    # When
    interface.on_validation_batch_end(
        {"loss": Tensor([2.0]), "predictions": Tensor([4.0, 5.0])},
        [
            (rand(10, 5), rand(10, 1), rand(5, 3), rand(5, 1), 1),
            (rand(10, 5), rand(10, 1), rand(5, 3), rand(5, 1), 0),
        ],
        0,
    )

    # Then
    assert len(interface.validation_predictions) == 1
    assert (interface.validation_predictions[0] == Tensor([4.0, 5.0])).all()
    assert len(interface.validation_labels) == 1
    assert (interface.validation_labels[0] == Tensor([1, 0])).all()


@patch("wsmf.metamodels.train.interface.TrainingInterface.calculate_loss")
def test_on_validation_epoch_end(calculate_loss_mock: Mock) -> None:
    # Given
    calculate_loss_mock.return_value = Tensor([0.0])
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))
    interface.validation_labels = [Tensor([1, 0]), Tensor([1, 1])]
    interface.validation_predictions = [Tensor([0.1, 0.2]), Tensor([0.3, 0.4])]

    # When
    interface.on_validation_epoch_end()

    # Then
    calculate_loss_mock.assert_called()
    calculate_loss_args = calculate_loss_mock.call_args
    assert (calculate_loss_args[0][0] == Tensor([1, 0, 1, 1])).all()
    assert (calculate_loss_args[0][1] == Tensor([0.1, 0.2, 0.3, 0.4])).all()


def test_extract_labels_and_similarities_from_batch() -> None:
    # Given
    batch = [
        (rand(10, 5), rand(10, 1), rand(5, 3), rand(5, 1), 1),
        (rand(10, 5), rand(10, 1), rand(5, 3), rand(5, 1), 0),
    ]
    interface = MockImplementation(OptimizerConfig(), lambda x, y: Tensor([0]))

    # When
    actual_labels, actual_similarities = (
        interface.extract_labels_and_similarities_from_batch(batch)
    )

    # Then
    assert (actual_labels == Tensor([1, 0])).all()
    assert (actual_similarities == Tensor([0.5, 0.5])).all()
