from typing import Any, Callable

from dataset2vec.config import OptimizerConfig
from torch import Tensor, rand
from torch.optim.optimizer import Optimizer

from wsmf.metamodels.train import LandmarkerReconstructionTrainingInterface


class MockImplementation(LandmarkerReconstructionTrainingInterface):
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


def test_extract_labels_and_similarities_from_batch() -> None:
    # Given
    batch = [
        (rand(10, 5), rand(10, 1), Tensor([4, 5, 6])),
        (rand(5, 3), rand(5, 1), Tensor([6, 5, 4])),
    ]
    implementation = MockImplementation(
        OptimizerConfig(), lambda x, y: Tensor([0])
    )

    # When
    labels, outputs = (
        implementation.extract_labels_and_similarities_from_batch(batch)
    )

    # Then
    assert len(labels) == 2
    assert (labels[0] == Tensor([4, 5, 6])).all()
    assert (labels[1] == Tensor([6, 5, 4])).all()
    assert len(outputs) == 2
    assert (outputs[0] == Tensor([1, 2, 3])).all()
    assert (outputs[1] == Tensor([2, 3, 4])).all()
