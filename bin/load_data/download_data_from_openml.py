import json
import warnings
from operator import itemgetter

import pytorch_lightning as pl
from loguru import logger
from openml import datasets, tasks
from tqdm import tqdm

from experiments_engine.data import (
    is_eligible_task,
    move_target_to_last_column,
)
from experiments_engine.paths import paths_provider

warnings.simplefilter("ignore")


def main() -> None:
    pl.seed_everything(123)
    logger.info("Loading tasks ids")
    with open(paths_provider.tasks_ids_path, "r") as f:
        tasks_ids = json.load(f)

    logger.info("Loading prohibited datasets names")
    with open(paths_provider.prohibited_datasets_path, "r") as f:
        prohibited_datasets = json.load(f)

    logger.info("Loading tasks")
    classification_tasks = tasks.list_tasks(
        task_type=tasks.TaskType.SUPERVISED_CLASSIFICATION
    )
    classification_tasks = list(
        map(
            itemgetter(1),
            filter(
                lambda item: item[0] in tasks_ids,
                classification_tasks.items(),
            ),
        )
    )
    classification_tasks = list(filter(is_eligible_task, classification_tasks))

    logger.info("Loading raw datasets")
    error_count = 0
    for task in (pbar := tqdm(classification_tasks)):
        pbar.set_postfix(
            {
                "features": task.get("NumberOfFeatures"),
                "instances": task.get("NumberOfInstances"),
                "classes": task.get("NumberOfClasses"),
                "errors": error_count,
            }
        )
        try:
            dataset = datasets.get_dataset(task["did"])
            if dataset.name in prohibited_datasets:
                continue
            dataset_df = dataset.get_data()[0]
            dataset_df = move_target_to_last_column(
                dataset_df, task["target_feature"]  # type: ignore
            )
            filename = task["name"]
            dataset_df.to_parquet(
                paths_provider.raw_datasets_path / f"{filename}.parquet",
                index=False,
            )
        except:  # noqa E722
            error_count += 1


if __name__ == "__main__":
    main()
