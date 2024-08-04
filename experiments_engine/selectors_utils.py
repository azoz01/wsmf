import json
from pathlib import Path
from typing import Tuple, Type

import pandas as pd
from dataset2vec.utils import DataUtils
from torch import Tensor

from experiments_engine.utils import extract_dataset_name_from_path
from wsmf.selectors.selector import WarmstartHpSelector

from .paths import paths_provider


def get_hp_selector_from_path(
    selector_cls: Type[WarmstartHpSelector],
    path: Path,
    configurations_path: Path,
    landmarkers_path: Path,
) -> WarmstartHpSelector:
    datasets_names = list(
        sorted([extract_dataset_name_from_path(p) for p in path.iterdir()])
    )
    datasets = {
        dataset_name: get_dataset_from_path_for_meta_train(
            paths_provider.train_meta_dataset_path / dataset_name
        )
        for dataset_name in datasets_names
    }
    datasets = dict(
        list(filter(lambda en: en[1][0].shape[0] <= 5_000, datasets.items()))
    )
    with open(landmarkers_path) as f:
        landmarkers = json.load(f)
    landmarkers = {
        dataset_name: Tensor(landmarkers[dataset_name]).cuda()
        for dataset_name in landmarkers.keys()
    }
    with open(configurations_path) as f:
        configurations = json.load(f)
    return selector_cls(
        metadataset=datasets,
        landmarkers=landmarkers,
        configurations=configurations,
    )


def get_dataset_from_path_for_meta_train(path: Path) -> Tuple[Tensor, Tensor]:
    dataset = pd.read_parquet(path / "test.parquet")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    pipeline = DataUtils.get_preprocessing_pipeline()
    X = pipeline.fit_transform(X)
    X = Tensor(X.values).cuda()
    y = Tensor(y.values).reshape(-1, 1).cuda()
    return X, y
