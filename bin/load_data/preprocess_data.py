import warnings

import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments_engine.data import (
    clean_and_binarize_classification,
    remove_unwanted_columns,
)
from experiments_engine.paths import paths_provider
from experiments_engine.utils import extract_dataset_name_from_path

warnings.simplefilter(action="ignore")


def main() -> None:
    pl.seed_everything(123)
    logger.info("Preprocessing & Saving tasks")
    for dataset_path in (
        pbar := tqdm(sorted(paths_provider.raw_datasets_path.iterdir()))
    ):
        df = pd.read_parquet(dataset_path)
        df = clean_and_binarize_classification(df)
        if (df.iloc[:, -1] == 1).sum() < 2 or (df.iloc[:, -1] == 0).sum() < 2:
            logger.warning(
                f"Skipping {dataset_path} due to too high class imbalance"
            )
            continue
        df = remove_unwanted_columns(df)
        df.to_parquet(
            paths_provider.datasets_binarized_path / dataset_path.name,
            index=False,
        )
        pbar.set_postfix(
            {
                "shape": df.shape,
                "p": df.iloc[:, -1].mean(),
                "task_name": extract_dataset_name_from_path(dataset_path),
            }
        )

    logger.info("Splitting datasets")
    for dataset_path in (
        pbar := tqdm(sorted(paths_provider.datasets_binarized_path.iterdir()))
    ):
        dataset_name = extract_dataset_name_from_path(dataset_path)
        pbar.set_postfix({"dataset": dataset_name})
        df = pd.read_parquet(dataset_path)
        df_train, df_test = train_test_split(
            df, stratify=df.iloc[:, -1], random_state=123
        )
        output_dataset_path = (
            paths_provider.datasets_splitted_path / dataset_name
        )
        output_dataset_path.mkdir(exist_ok=True, parents=True)
        df_train.to_parquet(output_dataset_path / "train.parquet", index=False)
        df_test.to_parquet(output_dataset_path / "test.parquet", index=False)

    logger.info("Finished")


if __name__ == "__main__":
    main()
