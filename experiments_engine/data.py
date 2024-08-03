from typing import Any

import numpy as np
import pandas as pd
from openml import OpenMLTask


def move_target_to_last_column(
    df: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    if df.columns[-1] == target_column:
        return df
    target = df[target_column]
    df = df.drop(columns=[target_column])
    df[target_column] = target
    return df


def is_eligible_task(task: OpenMLTask) -> Any:
    if task is None:
        return False
    if (
        task.get("NumberOfFeatures") is None  # type: ignore
        or task.get("NumberOfInstances") is None  # type: ignore
        or task.get("NumberOfClasses") is None  # type: ignore
    ):
        return False
    return (
        task.get("NumberOfFeatures") <= 50  # type: ignore
        and task.get("NumberOfInstances") <= 100_000  # type: ignore
        and task.get("NumberOfClasses") <= 10  # type: ignore
    )


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.tolist()[:-1]
    id_columns = filter(
        lambda c: c.lower().startswith("id") or c.lower().endswith("_id"),
        columns,
    )
    df = df.drop(columns=id_columns)
    df = df.select_dtypes(exclude=pd.Timestamp)
    return df


def clean_and_binarize_classification(df: pd.DataFrame) -> pd.DataFrame:
    target_variable = df.iloc[:, -1]
    df = df.loc[~target_variable.isna()]
    target_variable = target_variable[~target_variable.isna()]

    unique_categories = np.array(
        sorted(target_variable.drop_duplicates().tolist())
    )
    one_category_subset = np.random.uniform(size=len(unique_categories)) <= 0.5
    if one_category_subset.all() or (~one_category_subset).all():
        one_category_subset[-1] = not one_category_subset[-1]
    one_category_subset = unique_categories[one_category_subset]
    target_variable = target_variable.isin(one_category_subset).astype(int)
    target_name = df.columns[-1]
    df[target_name] = target_variable
    return df
