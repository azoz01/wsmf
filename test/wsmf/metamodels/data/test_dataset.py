import pytest
from torch import Tensor

from wsmf.metamodels.data import EncoderHpoDataset


def test_d2v_hpo_dataset_has_proper_length() -> None:
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

    # Then
    assert len(d2v_hpo_dataset) == 2


def test_d2v_hpo_dataset_has_proper_dataset_names() -> None:
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

    # Then
    assert d2v_hpo_dataset.dataset_names == ["dataset1", "dataset2"]


def test_d2v_hpo_dataset_returns_proper_data_on_index() -> None:
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
    actual_dataset_X, actual_dataset_y, actual_landmarkers = d2v_hpo_dataset[
        "dataset2"
    ]

    # Then
    assert (actual_dataset_X == dataset2_X).all()
    assert (actual_dataset_y == dataset2_y).all()
    assert (actual_landmarkers == Tensor([-1, -2, -3])).all()


def test_d2v_hpo_dataset_fail_when_inconsistent_data_sizes() -> None:
    # Given
    dataset1_X = Tensor([[1, 2, 3], [4, 5, 6]])
    dataset1_y = Tensor([[0], [1]])
    datasets = {
        "dataset1": (dataset1_X, dataset1_y),
    }
    landmarkers = {
        "dataset1": Tensor([1, 2, 3]),
        "dataset2": Tensor([-1, -2, -3]),
    }

    # Then
    with pytest.raises(AssertionError):
        EncoderHpoDataset(datasets, landmarkers)
