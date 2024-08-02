from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class EncoderHpoDataset(Dataset):

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

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, dataset_name) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            *self.datasets[dataset_name],
            self.hp_landmarkers[dataset_name],
        )

    @property
    def dataset_names(self):
        return list(sorted(self.datasets.keys()))
