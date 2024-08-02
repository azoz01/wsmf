import shutil

import numpy as np
import pytorch_lightning as pl
from loguru import logger
from tqdm import tqdm

from experiments_engine.paths import paths_provider
from experiments_engine.utils import extract_dataset_name_from_path


def main():
    pl.seed_everything(123)
    logger.info("Splitting to meta-train and meta-val")
    for dataset_path in tqdm(paths_provider.datasets_splitted_path.iterdir()):
        dataset_path_name = extract_dataset_name_from_path(dataset_path)
        if np.random.uniform() <= 0.3:
            shutil.copytree(
                paths_provider.datasets_splitted_path / dataset_path_name,
                paths_provider.val_meta_dataset_path / dataset_path_name,
            )
            shutil.copy(
                paths_provider.datasets_binarized_path
                / f"{dataset_path_name}.parquet",
                paths_provider.val_meta_dataset_path_for_plain_d2v
                / f"{dataset_path_name}.parquet",
            )
        else:
            shutil.copytree(
                paths_provider.datasets_splitted_path / dataset_path_name,
                paths_provider.train_meta_dataset_path / dataset_path_name,
            )
            shutil.copy(
                paths_provider.datasets_binarized_path
                / f"{dataset_path_name}.parquet",
                paths_provider.train_meta_dataset_path_for_plain_d2v
                / f"{dataset_path_name}.parquet",
            )


if __name__ == "__main__":
    main()
