"""Contains the functions related to the raw data refinement step."""
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Optional

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


def optimize_objects(df: pd.DataFrame, columns: Optional[List[str]] = None):
    """Infers the most efficient data type for object columns.

    Args:
        df (pd.DataFrame): The dataframe to optimize.
        columns (Optional[List[str]]): The columns to optimize. If None, all
            object columns are optimized.

    Returns:
        pd.DataFrame: The dataframe with optimized object columns.

    Example:
        >>> df = pd.DataFrame({
        ...     "a_num": [1, 2],
        ...     "b_categ": ["a", "b"],
        ...     "c_date": ["2020-01-01", "2020-01-02"],
        ... })
        >>> optimize_objects(df).dtypes
        a_num               int64
        b_categ          category
        c_date     datetime64[ns]
        dtype: object
    """
    df = df.copy()
    columns = columns or [col for col in df.columns if types.is_object_dtype(df[col])]
    for col in columns:
        series = df[col]
        if col.endswith("date"):
            series = pd.to_datetime(series)
        else:
            series = series.astype("category").cat.codes
        df[col] = series
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


def preprocess_bookings(df: pd.DataFrame, params: Dict[str, Any]):
    """Preprocesses the raw `hotel_bookings` dataset.

    1. Drops unnecessary columns.
    2. Remove rows where all columns are zero.
    3. Optimize categorical columns.
    4. Normalize numeric columns.
    5. Fill missing values.

    Args:
        df (pd.DataFrame): The raw `hotel_bookings` dataset.
        params (Dict[str, Any]): The parameters for preprocessing.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    df = (
        df.drop(columns=params["columns_to_drop"])
        .fillna(params.get("fillna", 0))
        .pipe(
            remove_if_all_equal,
            params["columns_to_validate"]["columns"],
            params["columns_to_validate"]["equal_to"],
        )
    )
    target = df[params["target"]].astype("int8")
    df = (
        df.drop(columns=[params["target"]])
        .pipe(log_normalize, params.get("columns_to_normalize", None))
        .pipe(optimize_objects, params.get("columns_to_optimize", None))
        .pipe(unpack_date, params["date_column"])
        .assign(**{params["target"]: target})
        .pipe(fillna, params.get("columns_to_fillna", {}))
    )
    return df
