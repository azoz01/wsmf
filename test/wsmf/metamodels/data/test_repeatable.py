from torch import Tensor

from wsmf.metamodels.data import (
    EncoderHpoDataset,
    EncoderMetricLearningLoader,
    GenericRepeatableDataLoader,
)


def test_encoder_metric_loader_calculates_sample_properly() -> None:
    # Given
    dataset1_X = Tensor([[1, 2, 3], [4, 5, 6]])
    dataset1_y = Tensor([[0], [1]])
    dataset2_X = Tensor([[7, 8, 9, 10], [4, 5, 6, 11]])
    dataset2_y = Tensor([[1], [0]])
    dataset3_X = Tensor([[7, 8, 9, 10, 11], [4, 5, 6, 11, 12]])
    dataset3_y = Tensor([[1], [0], [1]])
    datasets = {
        "dataset1": (dataset1_X, dataset1_y),
        "dataset2": (dataset2_X, dataset2_y),
        "dataset3": (dataset3_X, dataset3_y),
    }
    landmarkers = {
        "dataset1": Tensor([1, 2, 3]),
        "dataset2": Tensor([-1, -2, -3]),
        "dataset3": Tensor([-1, -1, -1]),
    }

    # When
    d2v_hpo_dataset = EncoderHpoDataset(datasets, landmarkers)
    dataloader = EncoderMetricLearningLoader(d2v_hpo_dataset, 2, 1)
    repeatable_loader = GenericRepeatableDataLoader(dataloader)
    batches1 = list(repeatable_loader)
    batches2 = list(repeatable_loader)

    # Then
    assert (batches1[0][0][0] == batches2[0][0][0]).all()
    assert (batches1[0][0][1] == batches2[0][0][1]).all()
    assert (batches1[0][0][2] == batches2[0][0][2]).all()
    assert (batches1[1][0][0] == batches2[1][0][0]).all()
    assert (batches1[1][0][1] == batches2[1][0][1]).all()
    assert (batches1[1][0][2] == batches2[1][0][2]).all()
