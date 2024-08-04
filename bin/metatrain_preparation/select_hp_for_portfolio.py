import argparse
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from optuna.trial import FixedTrial
from sklearn.cluster import KMeans
from tqdm import tqdm

import experiments_engine.hpo as hpo_cls_pkg
from experiments_engine.constants import N_CONFIGURATIONS_IN_PORTFOLIO
from experiments_engine.hpo import disable_optuna_logs
from experiments_engine.paths import paths_provider
from experiments_engine.portfolio_selection import (
    extract_best_configuration_idx_from_cluster_eval_results,
)
from experiments_engine.utils import extract_dataset_name_from_path, read_json

disable_optuna_logs()


def main():
    pl.seed_everything(123)
    logger.info("Parsing shell args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str)
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()

    logger.info("Calculating clusters of datasets")
    metafeatures_df = pd.read_parquet(paths_provider.metafeatures_path)
    considered_datasets = [
        extract_dataset_name_from_path(dataset_path)
        for dataset_path in paths_provider.train_meta_dataset_path.iterdir()  # noqa: E501
    ]
    metafeatures_df = metafeatures_df.loc[
        metafeatures_df["dataset_name"].isin(considered_datasets)
    ]
    metafeatures = metafeatures_df.iloc[:, :-1].values
    clusters = KMeans(
        n_clusters=N_CONFIGURATIONS_IN_PORTFOLIO, n_init=20
    ).fit_predict(metafeatures)

    logger.info("Performing tournament selection of best HP configurations")
    hp_portfolio_configurations = []
    objective_cls = getattr(hpo_cls_pkg, args.objective)
    datasets_names = metafeatures_df["dataset_name"]
    for cluster_id in (pbar := tqdm(range(N_CONFIGURATIONS_IN_PORTFOLIO))):
        cluster_mask = clusters == cluster_id
        datasets_from_cluster = sorted(datasets_names[cluster_mask].values)
        best_configurations_of_hp_from_dataset_cluster = [
            read_json(
                paths_provider.best_hpo_path
                / args.model_name
                / f"{dataset_name}.json"
            )
            for dataset_name in datasets_from_cluster
        ]
        datasets_data = [
            (
                pd.read_parquet(
                    paths_provider.train_meta_dataset_path
                    / dataset_name
                    / "train.parquet"
                ),
                pd.read_parquet(
                    paths_provider.train_meta_dataset_path
                    / dataset_name
                    / "test.parquet"
                ),
            )
            for dataset_name in datasets_from_cluster
        ]
        datasets_inside_clusters_performances = []
        for i, (df_train, df_test) in enumerate(datasets_data):
            objective = objective_cls(df_train, df_test)
            datasets_inside_clusters_performances.append([])
            for j, parameters in enumerate(
                best_configurations_of_hp_from_dataset_cluster
            ):
                metric = objective(FixedTrial(parameters))
                datasets_inside_clusters_performances[-1].append(metric)
                pbar.set_postfix_str(
                    f"dataset {i}/{len(datasets_data) - 1}, "
                    f"configuraion {j}/{len(best_configurations_of_hp_from_dataset_cluster) - 1}, "  # noqa
                    f"train shape {df_train.shape}, "
                    f"test shape {df_test.shape}"
                )
        datasets_inside_clusters_performances = np.array(
            datasets_inside_clusters_performances
        )
        best_configuration_idx = (
            extract_best_configuration_idx_from_cluster_eval_results(
                datasets_inside_clusters_performances
            )
        )

        best_hp_configuration_in_cluster = (
            best_configurations_of_hp_from_dataset_cluster[
                best_configuration_idx
            ]
        )
        hp_portfolio_configurations.append(best_hp_configuration_in_cluster)

    logger.info("Saving portfolio")
    with open(
        paths_provider.hp_portfolio_configurations_path
        / f"{args.model_name}.json",
        "w",
    ) as f:
        json.dump(hp_portfolio_configurations, f, indent=4)


if __name__ == "__main__":
    main()
