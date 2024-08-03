import gc
import json
from venv import logger

import pandas as pd
from tqdm import tqdm

from experiments_engine.hpo import (
    XGBoostObjective,
    get_best_study_params,
    perform_study,
)
from experiments_engine.paths import paths_provider
from experiments_engine.utils import extract_dataset_name_from_path


def main() -> None:
    objectives_to_evaluate = [XGBoostObjective]
    erroneous_datasets = []
    for dataset_path in (
        pbar := tqdm(
            list(sorted(paths_provider.datasets_splitted_path.iterdir()))
        )
    ):
        dataset_name = extract_dataset_name_from_path(dataset_path)
        df_train = pd.read_parquet(dataset_path / "train.parquet")
        df_test = pd.read_parquet(dataset_path / "test.parquet")
        pbar.set_postfix(
            {
                "dataset": dataset_name,
                "train_shape": df_train.shape,
                "test_shape": df_test.shape,
                "errors": len(erroneous_datasets),
            }
        )
        for objective_cls in objectives_to_evaluate:
            objective = objective_cls(df_train, df_test)
            best_params_objective_path = (
                paths_provider.best_hpo_path / objective.display_name
            )
            (best_params_objective_path).mkdir(exist_ok=True, parents=True)
            try:
                study = perform_study(
                    objective, f"{dataset_name}_{objective.display_name}"
                )
            except Exception as e:
                erroneous_datasets.append(
                    [objective.display_name, dataset_name]
                )
                logger.warning(f"Timeout for {dataset_name} - {e}")
            params = get_best_study_params(study)
            with open(
                best_params_objective_path / f"{dataset_name}.json",
                "w",
            ) as f:
                json.dump(params, f)
            gc.collect()
    logger.warning(f"Datasets with errors: {erroneous_datasets}")


if __name__ == "__main__":
    main()
