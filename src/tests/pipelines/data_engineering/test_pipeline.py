"""Tests everything related to data engineering pipeline."""
# pylint: disable=redefined-outer-name
import numpy as np
import pandas as pd
import pytest

from src.hotelbookingcancellation.pipelines.data_engineering.nodes import (
    _PreprocessBookingsParams,
    preprocess_bookings,
)
from src.hotelbookingcancellation.pipelines.data_engineering.pipeline import (
    create_pipeline,
)


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Dummy dataframe."""
    return pd.DataFrame(
        {
            "id0": [0, 1, 0, 2],
            "id1": [0, 1, 0, 2],
            "cat0": ["a", "b", "a", "c"],
            "cat1": ["a", "d", "e", "d"],
            "num0": [0.0, 1.0, 2.0, 3.0],
            "num1": [0.0, np.nan, np.nan, 3.0],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "drop": [0, 1, 2, 3],
            "t": [0, 1, 2, 3],
        }
    )


@pytest.fixture()
def min_params() -> _PreprocessBookingsParams:
    """Minimal parameters for preprocessing."""
    return {  # type: ignore
        "columns_to_drop": ["drop"],
        "fillna": 0,
        "columns_to_remove": {
            "columns": ["id0", "id1"],
            "equal_to": 0,
        },
        "date_column": "date",
        "target": "t",
    }


def test_preprocess_bookings_minimum(
    raw_df: pd.DataFrame, min_params: _PreprocessBookingsParams
):
    """Test preprocessing with minimal parameters."""
    df = preprocess_bookings(raw_df, min_params)
    assert "drop" not in df.columns
    assert df.isna().sum().sum() == 0
    assert len(df) == 2
    assert "year" in df.columns
    assert "month" in df.columns
    assert "day" in df.columns
    assert df["t"].dtype == np.int8
    assert df["num0"].tolist() != raw_df["num0"].iloc[[1, 3]].tolist()
    assert df["cat1"].dtype == np.int8


def test_preprocess_bookings_select_log_normalize(
    raw_df: pd.DataFrame, min_params: _PreprocessBookingsParams
):
    """Test preprocessing with selected log normalize columns."""
    params = min_params.copy()
    params["columns_to_normalize"] = ["num0"]
    df = preprocess_bookings(raw_df, params)
    assert df["num0"].tolist() != raw_df["num0"].iloc[[1, 3]].tolist()
    assert df["num1"].tolist() == [0.0, 3.0]


def test_preprocess_bookings_select_optimize(
    raw_df: pd.DataFrame, min_params: _PreprocessBookingsParams
):
    """Test preprocessing with selected optimize columns."""
    params = min_params.copy()
    params["columns_to_optimize"] = ["cat1"]
    df = preprocess_bookings(raw_df, params)
    assert df["cat1"].dtype == "int8"
    assert df["cat0"].dtype == "object"


def test_validate_pipeline_create():
    """Tests if a pipeline can be instantiated."""
    pipeline = create_pipeline()
    assert pipeline
