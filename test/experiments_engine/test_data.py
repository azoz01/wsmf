from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from experiments_engine.data import (
    clean_and_binarize_classification,
    move_target_to_last_column,
    remove_unwanted_columns,
)


def test_move_target_to_last_column_when_target_last() -> None:
    # Given
    df = pd.DataFrame({"col1": [1, 2], "col2": [1, 2]})

    # When
    actual_df = move_target_to_last_column(df, "col2")

    # Then
    assert (actual_df == df).all(axis=None)


def test_move_target_to_last_column_when_target_not_last() -> None:
    # Given
    df = pd.DataFrame({"col1": [1, 2], "col2": [1, 2]})

    # When
    actual_df = move_target_to_last_column(df, "col1")

    # Then
    assert (actual_df == df[["col2", "col1"]]).all(axis=None)


def test_remove_unwanted_columns() -> None:
    # Given
    df = pd.DataFrame({"id_1": [1], "2id": [2], "3_id": [3], "col": [4]})

    # When
    actual_df = remove_unwanted_columns(df)

    # Then
    assert (actual_df == df[["2id", "col"]]).all(axis=None)


def test_clean_and_binarize_classification_multiple_classes() -> None:
    # Given
    df = pd.DataFrame({"col": [1, 2, 3, 4], "target": [1, 2, 3, 1]})

    # When
    actual_df = clean_and_binarize_classification(df)

    # Then
    assert actual_df["target"].nunique() == 2
    assert actual_df["target"].min() == 0
    assert actual_df["target"].max() == 1


def test_clean_and_binarize_classification_text_classes() -> None:
    # Given
    df = pd.DataFrame({"col": [1, 2, 3, 4], "target": ["A", "B", "C", "A"]})

    # When
    actual_df = clean_and_binarize_classification(df)

    # Then
    assert actual_df["target"].nunique() == 2
    assert actual_df["target"].min() == 0
    assert actual_df["target"].max() == 1


@patch("numpy.random.uniform")
def test_clean_and_binarize_classification_when_subset_only_zeros(
    uniform_mock: Mock,
) -> None:
    uniform_mock.return_value = np.array([0.1, 0.2, 0.3])
    # Given
    df = pd.DataFrame({"col": [1, 2, 3, 4], "target": ["A", "B", "C", "A"]})

    # When
    actual_df = clean_and_binarize_classification(df)

    # Then
    assert actual_df["target"].nunique() == 2
    assert actual_df["target"].min() == 0
    assert actual_df["target"].max() == 1


@patch("numpy.random.uniform")
def test_clean_and_binarize_classification_when_subset_only_ones(
    uniform_mock: Mock,
) -> None:
    uniform_mock.return_value = np.array([0.6, 0.7, 0.8])
    # Given
    df = pd.DataFrame({"col": [1, 2, 3, 4], "target": ["A", "B", "C", "A"]})

    # When
    actual_df = clean_and_binarize_classification(df)

    # Then
    assert actual_df["target"].nunique() == 2
    assert actual_df["target"].min() == 0
    assert actual_df["target"].max() == 1
