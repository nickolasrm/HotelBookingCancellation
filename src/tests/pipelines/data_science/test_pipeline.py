"""Tests everything related to data engineering pipeline."""
# pylint: disable=redefined-outer-name
from typing import Tuple

import pandas as pd
import pytest
from catboost import CatBoostClassifier

from src.hotelbookingcancellation.pipelines.data_science.nodes import (
    evaluate,
    optimize,
    split_train_test,
)
from src.hotelbookingcancellation.pipelines.data_science.pipeline import create_pipeline


@pytest.fixture()
def df() -> pd.DataFrame:
    """Dummy dataframe."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "t": [1, 2, 1, 2, 1],
        }
    )


@pytest.fixture()
def train_test() -> Tuple[pd.DataFrame, ...]:
    """Train and test data."""
    return (
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
            }
        ),
        pd.DataFrame(
            {
                "a": [4, 5],
                "b": [4, 5],
            }
        ),
        pd.DataFrame(
            {
                "t": [1, 2, 1],
            }
        ),
        pd.DataFrame(
            {
                "t": [2, 1],
            }
        ),
    )


@pytest.fixture()
def model(train_test: Tuple[pd.DataFrame, ...]) -> CatBoostClassifier:
    """Fixture for the model."""
    return CatBoostClassifier(iterations=3, allow_writing_files=False).fit(
        train_test[0], train_test[2]
    )


def test_split_train_test(df: pd.DataFrame):
    """Test splitting the data into train and test."""
    x_train, x_test, y_train, y_test = split_train_test(
        df, {"target": "t", "test_size": 0.4}
    )
    assert len(x_train) == 3
    assert len(x_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2
    df_orig = (
        pd.concat([pd.concat([x_train, x_test]), pd.concat([y_train, y_test])], axis=1)
        .sort_values(by="a", ascending=True)
        .reset_index(drop=True)
    )
    assert df_orig.equals(df)


def test_optimize(train_test: Tuple[pd.DataFrame, ...]):
    """Test optimizing the model."""
    x_train, x_test, y_train, y_test = train_test
    model = optimize(x_train, y_train, {"iterations": 20, "allow_writing_files": False})
    assert model.get_param("iterations") == 20
    assert pytest.approx(model.predict(x_test), y_test)


def test_evaluate(train_test: Tuple[pd.DataFrame, ...], model: CatBoostClassifier):
    """Test the evaluation report of the model."""
    _, x_test, _, y_test = train_test
    report = evaluate(model, x_test, y_test, {"metrics": ["Accuracy"]})
    assert len(report["Accuracy"]) == 3
    assert all("step" in el and "value" in el for el in report["Accuracy"])


def test_validate_pipeline_create():
    """Tests if a pipeline can be instantiated."""
    pipeline = create_pipeline()
    assert pipeline
