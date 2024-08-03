from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class EncoderHpoDataset(Dataset):  # type: ignore
    """
    Dataset class for encoding HPO data.

    Combines datasets and corresponding HPO landmarkers into a single dataset.

    Parameters
    ----------
    datasets : dict[str, Tuple[Tensor, Tensor]]
        Dictionary of datasets, where keys are dataset names and values are
        tuples of tensors - features and targets.
    hp_landmarkers : dict[str, Tensor]
        Dictionary of HPO landmarkers, where keys are dataset names and
        values are tensors.

    Attributes
    ----------
    datasets : dict[str, Tuple[Tensor, Tensor]]
        Stored datasets.
    hp_landmarkers : dict[str, Tensor]
        Stored HPO landmarkers.

    Methods
    -------
    __len__()
        Returns the number of datasets.
    __getitem__(dataset_name)
        Returns a tuple of tensors - features and targets for the specified
        dataset with HPO landamrkers.
    """

    def __init__(
        self,
        datasets: dict[str, Tuple[Tensor, Tensor]],
        hp_landmarkers: dict[str, Tensor],
    ):
        self.datasets = datasets
        self.hp_landmarkers = hp_landmarkers

        assert list(sorted(datasets.keys())) == list(
            sorted(hp_landmarkers.keys())
        ), "Datasets and landmarkers should have same number of entries"

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, dataset_name: str) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            *self.datasets[dataset_name],
            self.hp_landmarkers[dataset_name],
        )

    @property
    def dataset_names(self) -> list[str]:
        return list(sorted(self.datasets.keys()))
