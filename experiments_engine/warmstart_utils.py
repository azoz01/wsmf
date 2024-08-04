from pathlib import Path
from typing import Tuple, Type

import optuna
import pandas as pd
from optuna.samplers import BaseSampler, TPESampler
from torch import Tensor

from experiments_engine.hpo import Objective, disable_optuna_logs
from wsmf.selectors import LandmarkerHpSelector
from wsmf.selectors.selector import WarmstartHpSelector


def get_hpo_task_from_path(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(path / "train.parquet"),
        pd.read_parquet(path / "test.parquet"),
    )


def perform_ground_truth_warm_start_experiment(
    objective: Objective,
    objective_landmarkers: Tensor,
    warm_starter: LandmarkerHpSelector,
    n_trials: int = 30,
    n_initial_trials: int = 10,
    seed: int = 1,
    sampler_cls: Type[BaseSampler] = TPESampler,
) -> pd.DataFrame:
    disable_optuna_logs()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler_cls(  # type: ignore
            seed=seed,
            n_startup_trials=n_initial_trials,
        ),
    )
    if n_initial_trials > 0:
        initial_trials = warm_starter.propose_configurations(
            objective_landmarkers,  # type: ignore
            n_initial_trials,
        )
        for trial in initial_trials:
            study.enqueue_trial(trial)
    study.optimize(objective, n_trials)
    return study.trials_dataframe()


def perform_warm_start_experiment(
    objective: Objective,
    warm_starter: WarmstartHpSelector | None,
    n_trials: int = 30,
    n_initial_trials: int = 10,
    seed: int = 1,
    sampler_cls: Type[BaseSampler] = TPESampler,
) -> pd.DataFrame:
    disable_optuna_logs()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler_cls(  # type: ignore
            seed=seed, n_startup_trials=n_initial_trials
        ),
    )
    if n_initial_trials > 0 and warm_starter is not None:
        X_warm_start, y_warm_start = (
            Tensor(objective.X_test.values).cuda(),
            Tensor(objective.y_test.values).reshape(-1, 1).cuda(),
        )
        initial_trials = warm_starter.propose_configurations(
            (X_warm_start, y_warm_start), n_initial_trials
        )
        for trial in initial_trials:
            study.enqueue_trial(trial)
    study.optimize(objective, n_trials)
    return study.trials_dataframe()
