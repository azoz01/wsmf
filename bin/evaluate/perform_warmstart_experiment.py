import argparse
import json
import warnings

import pandas as pd
import pytorch_lightning as pl
import yaml
from loguru import logger
from optuna import samplers
from torch import Tensor
from tqdm import tqdm

import experiments_engine.hpo as hpo_cls_pkg
from experiments_engine.hp_selectors.baselines import LandmarkerHpSelector
from experiments_engine.hp_selectors.factory import (
    SelectorsFactory,
    get_hp_selector_from_path,
)
from experiments_engine.paths import paths_provider
from experiments_engine.utils import extract_dataset_name_from_path
from experiments_engine.warmstart_utils import (
    get_hpo_task_from_path,
    perform_ground_truth_warm_start_experiment,
    perform_warm_start_experiment,
)

warnings.simplefilter("ignore")


def main():
    pl.seed_everything(123)
    logger.info("Parsing shell args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--sampler-name", type=str)
    args = parser.parse_args()
    with open(paths_provider.hp_selectors_path, "r") as f:
        config = yaml.load(f, yaml.CLoader)
    objective_cls = getattr(hpo_cls_pkg, args.objective)
    sampler_cls = getattr(samplers, args.sampler_name)

    logger.info("Initializing configurations selectors")
    selectors = [
        (
            config_entry.pop("name"),
            SelectorsFactory.get_selector_from_config(
                config_entry, args.model_name
            ),
        )
        for config_entry in config
    ]
    landmarker_based_selector = get_hp_selector_from_path(
        LandmarkerHpSelector,
        paths_provider.train_meta_dataset_path,
        paths_provider.hp_portfolio_configuratioons_path
        / f"{args.model_name}_half_random.json",
        paths_provider.landmarkers_path / f"{args.model_name}.json",
    )

    with open(
        paths_provider.landmarkers_path / f"{args.model_name}.json"
    ) as f:
        landmarkers_all = json.load(f)

    logger.info("Starting computation")
    experiment_results = []
    for dataset_path in (
        pbar := tqdm(
            list(sorted(paths_provider.val_meta_dataset_path.iterdir()))
        )
    ):
        objective = objective_cls(*get_hpo_task_from_path(dataset_path))
        pbar.set_postfix_str(extract_dataset_name_from_path(dataset_path))
        hp_result = perform_ground_truth_warm_start_experiment(
            objective,
            Tensor(
                landmarkers_all[extract_dataset_name_from_path(dataset_path)]
            ).cuda(),
            landmarker_based_selector,  # type: ignore
            seed=1,
            n_trials=20,
            n_initial_trials=5,
            sampler_cls=sampler_cls,
        )
        hp_result["dataset"] = extract_dataset_name_from_path(dataset_path)
        hp_result["warmstart"] = "Landmarkers"
        experiment_results.append(hp_result)

        for selector_name, selector in selectors:
            hp_result = perform_warm_start_experiment(
                objective,
                selector,
                seed=1,
                n_trials=20,
                n_initial_trials=5,
                sampler_cls=sampler_cls,
            )
            hp_result["dataset"] = extract_dataset_name_from_path(dataset_path)
            hp_result["warmstart"] = selector_name
            experiment_results.append(hp_result)

    logger.info("Postprocessing results")
    experiment_results = pd.concat(experiment_results, axis=0).reset_index(
        drop=True
    )
    logger.info("Saving results")
    experiment_results.to_csv(
        paths_provider.warmstart_results_path / f"{args.model_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
