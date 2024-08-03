from unittest.mock import Mock, patch

import numpy as np
from torch import Tensor

from wsmf.metamodels.data import EncoderHpoDataset, EncoderMetricLearningLoader


@patch("numpy.random.choice")
def test_encoder_metric_loader_calculates_sample_properly(
    choice_mock: Mock,
) -> None:
    # Given
    choice_mock.return_value = [0, 1]
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
    dataloader = EncoderMetricLearningLoader(d2v_hpo_dataset, 1, 1)
    sample = list(dataloader)[0][0]

    # Then
    assert (sample[0] == dataset1_X).all()
    assert (sample[1] == dataset1_y).all()
    assert (sample[2] == dataset2_X).all()
    assert (sample[3] == dataset2_y).all()
    assert np.isclose(sample[4], 56 / 3)


def test_encoder_metric_loader_returns_proper_number_of_batches() -> None:
    # Given
    dataset1_X = Tensor([[1, 2, 3], [4, 5, 6]])
    dataset1_y = Tensor([[0], [1]])
    dataset2_X = Tensor([[7, 8, 9, 10], [4, 5, 6, 11]])
    dataset2_y = Tensor([[1], [0]])
    datasets = {
        "dataset1": (dataset1_X, dataset1_y),
        "dataset2": (dataset2_X, dataset2_y),
    }
    landmarkers = {
        "dataset1": Tensor([1, 2, 3]),
        "dataset2": Tensor([-1, -2, -3]),
    }

    # When
    d2v_hpo_dataset = EncoderHpoDataset(datasets, landmarkers)
    dataloader = EncoderMetricLearningLoader(d2v_hpo_dataset, 16, 32)
    batches = list(dataloader)

    # Then
    assert len(batches) == 16


def test_encoder_metric_loader_returns_batch_with_proper_size() -> None:
    # Given
    dataset1_X = Tensor([[1, 2, 3], [4, 5, 6]])
    dataset1_y = Tensor([[0], [1]])
    dataset2_X = Tensor([[7, 8, 9, 10], [4, 5, 6, 11]])
    dataset2_y = Tensor([[1], [0]])
    datasets = {
        "dataset1": (dataset1_X, dataset1_y),
        "dataset2": (dataset2_X, dataset2_y),
    }
    landmarkers = {
        "dataset1": Tensor([1, 2, 3]),
        "dataset2": Tensor([-1, -2, -3]),
    }

    # When
    d2v_hpo_dataset = EncoderHpoDataset(datasets, landmarkers)
    dataloader = EncoderMetricLearningLoader(d2v_hpo_dataset, 16, 32)
    batch = list(dataloader)[0]

    # Then
    assert len(batch) == 32
