from typing import Any, Callable

from dataset2vec.config import OptimizerConfig
from torch import Tensor, rand
from torch.optim.optimizer import Optimizer

from wsmf.metamodels.train import MetricLearningTrainingInterface


class MockImplementation(MetricLearningTrainingInterface):
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        loss_function: Callable[[Tensor, Tensor], Tensor],
    ):
        super().__init__(optimizer_config, loss_function)
        mock_encodings = [
            Tensor([1, 2, 3]),
            Tensor([2, 3, 4]),
        ]
        self.mock_encodings_generator = (pred for pred in mock_encodings)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        return next(self.mock_encodings_generator)

    def configure_optimizers(  # type: ignore
        self,
    ) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        return None  # type: ignore


def test_calculate_loss() -> None:
    # Given
    implementation = MockImplementation(
        OptimizerConfig(), lambda x, y: Tensor([0])
    )

    # When
    actual_similarity = implementation.calculate_datasets_similarity(
        rand((10, 5)), rand((10, 1)), rand((5, 3)), rand((5, 1))
    )

    # Then
    assert (actual_similarity == Tensor([1.0])).all()
