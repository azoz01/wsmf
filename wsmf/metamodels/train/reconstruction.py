from torch import Tensor, stack

from .interface import TrainingInterface


class LandmarkerReconstructionTrainingInterface(TrainingInterface):
    """
    Base class for landmarker reconstruction training based meta-models.
    """

    def extract_labels_and_similarities_from_batch(
        self, batch: list[tuple[Tensor, Tensor, Tensor]]  # type: ignore
    ) -> tuple[Tensor, Tensor]:
        outputs = []
        labels = []
        for X, y, label in batch:
            outputs.append(self(X, y))
            labels.append(label)
        return stack(labels), stack(outputs)

    def calculate_datasets_similarity(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        raise NotImplementedError(
            "This training interface does not calculate similarities"
        )
