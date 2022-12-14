"""Contains the functions related to the raw data refinement step."""
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from pandas.api import types


def remove_if_all_equal(df: pd.DataFrame, columns: List[str], value: Any):
    """Removes all rows where all specified columns contain zero.

    Args:
        df (pd.DataFrame): The dataframe to remove rows from.
        columns (List[str]): The columns to check for zero values.

    Returns:
        pd.DataFrame: The dataframe with the rows removed.

    Example:
        >>> df = pd.DataFrame({"a": [0, 1, 2], "b": [0, 0, 0]})
        >>> remove_if_all_equal(df, ["a", "b"], 0)
           a  b
        1  1  0
        2  2  0
    """
    ghosts = reduce(lambda acc, col: (df[col] == value) & acc, columns, True)
    return df[~ghosts]


def log_normalize(df: pd.DataFrame, columns: Optional[List[str]] = None):
    """Normalizes numeric columns with log+1.

    Args:
        df (pd.DataFrame): The dataframe to normalize.
        columns (Optional[List[str]]): The columns to normalize. If None, all
            numerical columns are normalized.

    Returns:
        pd.DataFrame: The dataframe with normalized columns.

    Example:
        >>> df = pd.DataFrame({"a": [0, 1, 2], "b": [0, 0, 0]})
        >>> log_normalize(df, ["a", "b"])
                  a    b
        0  0.000000  0.0
        1  0.693147  0.0
        2  1.098612  0.0
    """
    df = df.copy()
    columns = columns or [col for col in df.columns if types.is_numeric_dtype(df[col])]
    for col in columns:
        df[col] = np.log(df[col] + 1)
    return df


def map_columns(df: pd.DataFrame, mappings: Dict[str, Dict[str, int]]):
    """Maps values in columns to integers.

    Args:
        df (pd.DataFrame): The dataframe to map values in.
        mappings (Dict[str, Dict[str, int]]): The mappings to apply.

    Returns:
        pd.DataFrame: The dataframe with mapped values.

    Example:
        >>> df = pd.DataFrame({"a": ["a", "b", "c"], "b": ["a", "a", "b"]})
        >>> map_columns(df, {"a": {"a": 0, "b": 1, "c": 2}, "b": {"a": 0, "b": 1}})
           a  b
        0  0  0
        1  1  0
        2  2  1
    """
    df = df.copy()
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)
    return df


def unpack_date(df: pd.DataFrame, column: str):
    """Unpacks a date column into multiple columns.

    Args:
        df (pd.DataFrame): The dataframe to unpack.
        column (str): The column to unpack.

    Returns:
        pd.DataFrame: The dataframe with unpacked columns.

    Example:
        >>> df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"]})
        >>> unpack_date(df, "date")
           year  month  day
        0  2020      1    1
        1  2020      1    2
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    df["year"] = df[column].dt.year
    df["month"] = df[column].dt.month
    df["day"] = df[column].dt.day
    df = df.drop(columns=[column])
    return df


FILLNA_FNS: Dict[str, Callable[[pd.Series], Any]] = {
    "mean": np.mean,
    "median": np.median,
}


def fillna(df: pd.DataFrame, columns: Dict[str, Literal["mean", "median"]]):
    """Fill missing values a constant value from `function`.

    Args:
        df (pd.DataFrame): The dataframe to fill.
        columns (Dict[str, Literal["mean", "median"]]): The columns to fill.
            The value is the function to use to fill the missing values.

    Returns:
        pd.DataFrame: The dataframe with filled columns.

    Example:
        >>> df = pd.DataFrame({"a": [0.0, np.nan, 2.0], "b": [np.nan, 0, 0]})
        >>> fillna(df, {"a": "mean"})
             a    b
        0  0.0  NaN
        1  1.0  0.0
        2  2.0  0.0
    """
    df = df.copy()
    for col, fn in columns.items():
        df[col] = df[col].fillna(FILLNA_FNS[fn](df[col]))
    return df


class _ColumnsToRemove(TypedDict):
    """The columns to validate."""

    columns: List[str]
    """Columns to check."""
    equal_to: Any
    """Value to match."""


class _PreprocessBookingsParams(TypedDict, total=False):
    columns_to_drop: List[str]
    """Name of the dataframe columns to drop."""
    fillna: Any
    """Value to replace nans with."""
    columns_to_remove: _ColumnsToRemove
    """Columns to remove if all values are equal to `equal_to`."""
    target: str
    """Target column name."""
    date_column: str
    """Date column name."""
    columns_to_normalize: Optional[List[str]]
    """Columns to normalize with log+1. If None, all numerical columns are selected."""
    columns_to_optimize: Optional[List[str]]
    """Columns to optimize with `optimize_objects`. If None, all object columns are
    selected."""
    columns_to_fillna: Dict[str, Literal["mean", "median"]]
    """Columns to fill missing values with `fillna`."""


def preprocess_bookings(df: pd.DataFrame, params: _PreprocessBookingsParams):
    """Preprocesses the raw `hotel_bookings` dataset.

    1. Drops unnecessary columns.
    2. Remove rows where all columns are zero.
    3. Maps categorical columns.
    4. Normalize numeric columns.
    5. Fill missing values.

    Args:
        df (pd.DataFrame): The raw `hotel_bookings` dataset.
        params (_PreprocessBookingsParams): The parameters for preprocessing.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    df = (
        df.drop(columns=params["columns_to_drop"], errors="ignore")
        .fillna(params.get("fillna", 0))
        .pipe(
            remove_if_all_equal,
            params["columns_to_remove"]["columns"],
            params["columns_to_remove"]["equal_to"],
        )
    )
    target = df[params["target"]].astype("int8")
    df = (
        df.drop(columns=[params["target"]])
        .pipe(log_normalize, params.get("columns_to_normalize", None))
        .pipe(map_columns, params.get("columns_to_map", {}))
        .pipe(unpack_date, params["date_column"])
        .assign(**{params["target"]: target})
        .pipe(fillna, params.get("columns_to_fillna", {}))
    )
    return df
