from abc import ABC, abstractmethod
from typing import Any, Callable

import optuna
import pandas as pd
from dataset2vec.utils import DataUtils
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


class Objective(ABC):

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        metric: Callable[[NDArray[Any], NDArray[Any]], float] = roc_auc_score,
    ):
        self.X_train, self.y_train = (
            df_train.iloc[:, :-1],
            df_train.iloc[:, -1],
        )
        self.X_test, self.y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
        self.pipeline = DataUtils.get_preprocessing_pipeline()
        self.X_train = self.pipeline.fit_transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)
        self.metric = metric

    def __call__(self, trial: optuna.Trial) -> float:
        X_train, y_train, X_test, y_test = self.get_data()
        model = self.get_model(trial)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]
        return self.metric(y_test, probas)

    @abstractmethod
    def get_model(self, trial: optuna.Trial) -> BaseEstimator:
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        pass

    def get_data(self) -> tuple[Any, Any, Any, Any]:
        return self.X_train, self.y_train, self.X_test, self.y_test


class XGBoostObjective(Objective):
    def get_model(self, trial: optuna.Trial) -> BaseEstimator:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 1000),
            eta=trial.suggest_float("eta", 1e-5, 1, log=True),
            gamma=trial.suggest_float("gamma", 1e-5, 1000, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1e-5, 100, log=True
            ),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 1000, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 1000, log=True),
            n_jobs=12,
        )

    @property
    def display_name(self) -> str:
        return "xgboost"


def perform_study(
    objective: Callable[[optuna.Trial], float],
    study_name: str,
    n_trials: int = 50,
    direction: str = "maximize",
) -> optuna.Study:
    disable_optuna_logs()
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=1),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def get_best_study_params(study: optuna.Study) -> dict[str, Any]:
    return study.best_params


def disable_optuna_logs() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
