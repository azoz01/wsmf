import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from dataset2vec.utils import DataUtils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from experiments_engine.paths import paths_provider

from .utils import extract_dataset_name_from_path


def get_dataset_from_path(path: Path) -> Tuple[Tensor, Tensor]:
    dataset = pd.read_parquet(path / "test.parquet")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    pipeline = DataUtils.get_preprocessing_pipeline()
    X = pipeline.fit_transform(X)
    X = Tensor(X.values).cuda()
    y = Tensor(y.values).reshape(-1, 1).cuda()
    return X, y


def load_datasets_with_landmarkers(
    reduce_landmarkers_dimensionality: bool = False,
) -> Tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_datasets_names = list(
        sorted(
            [
                extract_dataset_name_from_path(p)
                for p in paths_provider.train_meta_dataset_path.iterdir()  # noqa: E501
            ]
        )
    )
    val_datasets_names = list(
        sorted(
            [
                extract_dataset_name_from_path(p)
                for p in paths_provider.val_meta_dataset_path.iterdir()
            ]
        )
    )
    train_datasets = {
        dataset_name: get_dataset_from_path(
            paths_provider.train_meta_dataset_path / dataset_name
        )
        for dataset_name in train_datasets_names
    }
    train_datasets = dict(
        list(
            filter(
                lambda en: en[1][0].shape[0] <= 5_000, train_datasets.items()
            )
        )
    )

    val_datasets = {
        dataset_name: get_dataset_from_path(
            paths_provider.val_meta_dataset_path / dataset_name
        )
        for dataset_name in val_datasets_names
    }
    val_datasets = dict(
        list(
            filter(lambda en: en[1][0].shape[0] <= 5_000, val_datasets.items())
        )
    )

    with open(paths_provider.landmarkers_path / "xgboost.json") as f:
        landmarkers = json.load(f)

    train_landmarkers = {
        dataset_name: Tensor(landmarkers[dataset_name]).cuda()
        for dataset_name in train_datasets.keys()
    }
    val_landmarkers = {
        dataset_name: Tensor(landmarkers[dataset_name]).cuda()
        for dataset_name in val_datasets.keys()
    }
    if reduce_landmarkers_dimensionality:
        train_landmarkers, val_landmarkers = (
            __project_landmarkers_to_smaller_space(
                train_landmarkers, val_landmarkers
            )
        )
    return train_datasets, train_landmarkers, val_datasets, val_landmarkers


def __project_landmarkers_to_smaller_space(
    train_landmarkers: dict[str, Tensor], val_landmarkers: dict[str, Tensor]
) -> Tuple[dict[str, Tensor], dict[str, Tensor]]:
    projection_train_data = np.stack(
        list([item.cpu().numpy() for item in train_landmarkers.values()])
    )
    projection = PCA(n_components=3).fit(projection_train_data)
    scaling_train_data = projection.transform(projection_train_data)
    scaling = StandardScaler().fit(scaling_train_data)
    out_train_landmarkers = {
        name: Tensor(
            scaling.transform(
                projection.transform(value.cpu().numpy().reshape(1, -1))
            )[0]
        )
        for name, value in train_landmarkers.items()
    }
    out_val_landmarkers = {
        name: Tensor(
            scaling.transform(
                projection.transform(value.cpu().numpy().reshape(1, -1))
            )[0]
        )
        for name, value in val_landmarkers.items()
    }
    return out_train_landmarkers, out_val_landmarkers
