from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from numpy import random
from sklearn.datasets import make_classification
from tqdm import tqdm

from experiments_engine.paths import paths_provider


def generate_random_dataset() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = random.randint(1_000, 20_000)
    n_features = random.randint(2, 50)
    n_informative = random.randint(2, n_features + 1)
    one_class_weight = random.uniform()
    weights = [1 - one_class_weight, one_class_weight]
    flip_y = random.uniform(0, 0.5)
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        weights=weights,
        flip_y=flip_y,
    )


def generate_random_number_of_datasets() -> list[pd.DataFrame]:
    X, y = generate_random_dataset()
    y = y.reshape(-1, 1)
    df = pd.DataFrame(data=np.concatenate([X, y], axis=1))
    n_datasets = np.random.randint(1, 6)
    if n_datasets == 1:
        return [df]
    else:
        dataset_size = df.shape[0] // n_datasets
        dfs = [
            df.iloc[
                i
                * dataset_size : np.min(  # noqa: E203
                    [(i + 1) * dataset_size, df.shape[0]]
                )
            ]
            for i in range(n_datasets)
        ]
        return dfs


def main() -> None:
    pl.seed_everything(123)

    logger.info("Generating synthetic datasets")
    dataset_counter = 0
    for _ in tqdm(range(200)):
        dfs = generate_random_number_of_datasets()
        for df in dfs:
            df.to_parquet(
                paths_provider.raw_datasets_path
                / f"{dataset_counter:04d}.parquet",
                index=False,
            )
            dataset_counter += 1


if __name__ == "__main__":
    main()
