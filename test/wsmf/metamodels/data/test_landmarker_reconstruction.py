from unittest.mock import Mock, patch

from torch import Tensor

from wsmf.metamodels.data import (
    EncoderHpoDataset,
    LandmarkerReconstructionLoader,
)


@patch("numpy.random.choice")
def test_landmarker_reconstruction_loader_returns_proper_sample(
    choice_mock: Mock,
) -> None:
    # Given
    choice_mock.return_value = [0, 1]
    dataset1_X = Tensor([[1, 2, 3], [4, 5, 6]])
    dataset1_y = Tensor([[0], [1]])
    datasets = {
        "dataset1": (dataset1_X, dataset1_y),
    }
    landmarkers = {
        "dataset1": Tensor([1, 2, 3]),
    }

    # When
    d2v_hpo_dataset = EncoderHpoDataset(datasets, landmarkers)
    dataloader = LandmarkerReconstructionLoader(d2v_hpo_dataset, 1)
    sample = list(dataloader)[0][0]

    # Then
    assert (sample[0] == dataset1_X).all()
    assert (sample[1] == dataset1_y).all()
    assert (sample[2] == landmarkers["dataset1"]).all()  # type: ignore


def test_landmarker_reconstruction_loader_returns_proper_batch_size() -> None:
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
    dataloader = LandmarkerReconstructionLoader(d2v_hpo_dataset, 2)
    batch = list(dataloader)[0]

    # Then
    assert len(batch) == 2


def test_landmarker_reconstruction_loader_returns_all_datasets() -> None:
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
    dataloader = LandmarkerReconstructionLoader(d2v_hpo_dataset, 1)
    returned_datasets = list(dataloader)

    # Then
    assert (returned_datasets[0][0][0] == dataset1_X).all()
    assert (returned_datasets[0][0][1] == dataset1_y).all()
    assert (returned_datasets[0][0][2] == landmarkers["dataset1"]).all()  # type: ignore # noqa E501
    assert (returned_datasets[1][0][0] == dataset2_X).all()
    assert (returned_datasets[1][0][1] == dataset2_y).all()
    assert (returned_datasets[1][0][2] == landmarkers["dataset2"]).all()  # type: ignore # noqa E501
