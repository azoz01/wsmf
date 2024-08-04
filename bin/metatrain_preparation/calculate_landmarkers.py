import argparse
import json

import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from optuna.trial import FixedTrial
from tqdm import tqdm

import experiments_engine.hpo as hpo_cls_pkg
from experiments_engine.paths import paths_provider
from experiments_engine.utils import extract_dataset_name_from_path


def main():
    pl.seed_everything(123)
    logger.info("Parsing shell args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str)
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()
    objective_cls = getattr(hpo_cls_pkg, args.objective)

    logger.info("Loading portfolio configurations")
    with open(
        paths_provider.hp_portfolio_configurations_path
        / f"{args.model_name}.json"
    ) as f:
        portfolio = json.load(f)

    logger.info("Calculating landmarkers")
    landmarkers_output = dict()
    for dataset_path in (
        pbar := tqdm(
            list(sorted(paths_provider.datasets_splitted_path.iterdir()))
        )
    ):
        dataset_name = extract_dataset_name_from_path(dataset_path)
        df_train = pd.read_parquet(dataset_path / "train.parquet")
        df_test = pd.read_parquet(dataset_path / "test.parquet")
        objective = objective_cls(df_train, df_test)
        landmarkers = []
        for i, params in enumerate(portfolio):
            landmarkers.append(objective(FixedTrial(params)))
            pbar.set_postfix_str(
                "Evaluating configuration "
                f"{i}/{len(portfolio)} on {dataset_name}"
            )
        landmarkers_output[dataset_name] = landmarkers

    logger.info("Saving landmarkers")
    with open(
        paths_provider.landmarkers_path / f"{args.model_name}.json", "w"
    ) as f:
        json.dump(landmarkers_output, f, indent=4)


if __name__ == "__main__":
    main()
