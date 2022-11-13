"""Contains functions related to the data science step."""
import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore


# Created this function in order to not require sklearn's train_test_split as a
# dependency
def index_split(
    *dfs: pd.DataFrame, ratio: float
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Splits the length equal dataframes into two given a ratio.

    Args:
        *dfs (pd.DataFrame): The dataframes to split.
        ratio (float): Proportion of the first split.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: The splitted dataframes.

    Example:
        >>> np.random.seed(10)
        >>> dfs = index_split(pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6]),
        ...                   ratio=0.5)
        >>> dfs[0][0]
           0
        0  1
        >>> dfs[0][1]
           0
        0  2
        1  3
        >>> dfs[1][0]
           0
        0  4
        >>> dfs[1][1]
           0
        0  5
        1  6
    """
    indexes = np.arange(len(dfs[0]))
    filter_ = np.random.choice(indexes, size=int(ratio * len(indexes)), replace=False)
    not_filter = np.setdiff1d(indexes, filter_, assume_unique=True)
    return [
        (
            df.iloc[filter_].reset_index(drop=True),
            df.iloc[not_filter].reset_index(drop=True),
        )
        for df in dfs
    ]


def split_train_test(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and test sets.

    Args:
        df (pd.DataFrame): The input data.
        params (Dict[str, Any]): The parameters.
            * test_size (float): The size of the test set.
            * target (str): The target column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            0. x_train (pd.DataFrame): The training features.
            1. x_test (pd.DataFrame): The test features.
            2. y_train (pd.DataFrame): The training target.
            3. y_test (pd.DataFrame): The test target.
    """
    train_size = 1 - params["test_size"]
    target = params["target"]
    parts = index_split(
        df.drop(columns=[target]), df[target].to_frame(), ratio=train_size
    )
    return tuple(itertools.chain(*parts))  # type: ignore


def optimize(
    x: pd.DataFrame, y: pd.DataFrame, params: Dict[str, Any]
) -> CatBoostClassifier:
    """Generates a `CatBoostClassifier` model.

    Note:
        From all the possible algorithms tested in the optimization process,
        `CatBoostClassifier` seemed to be the best choice since it has the best
        performance (highest accuracy, precision, recall and f1-score) and its
        a suitable algorithm for categorical data like the one we have.

    Args:
        x (pd.DataFrame): The training features.
        y (pd.DataFrame): The training target.
        params (Dict[str, Any]): Kwargs for the `CatBoostClassifier`.

    Returns:
        CatBoostClassifier: The trained model.
    """
    params["train_dir"] = params.get("train_dir", "logs/catboost")
    cat = CatBoostClassifier(**params)
    cat.fit(x, y)
    return cat
