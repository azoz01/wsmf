from torch import Tensor

from .interface import TrainingInterface


class MetricLearningTrainingInterface(TrainingInterface):
    """
    Base class for metric learning based meta-models.
    """

    def calculate_datasets_similarity(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        emb1 = self(X1, y1)
        emb2 = self(X2, y2)
        return ((emb1 - emb2) ** 2).mean()  # type: ignore
