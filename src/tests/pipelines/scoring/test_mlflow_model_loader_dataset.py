"""Tests for the `MlflowModelLoaderDataSet` class."""
# pylint: disable=redefined-outer-name,unused-argument,pointless-statement
import time

import mlflow
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from kedro.io import DataSetError
from pytest_mock import MockFixture

from src.hotelbookingcancellation.pipelines.scoring import MlflowModelLoaderDataSet


@pytest.fixture
def model() -> CatBoostClassifier:
    """Creates a `CatBoost` model."""
    return CatBoostClassifier(iterations=1).fit(
        pd.DataFrame({"a": [1, 2, 3]}),
        pd.DataFrame({"b": [1, 2, 3]}),
    )


@pytest.fixture()
def model_name(setup_mlflow, model: CatBoostClassifier) -> str:
    """Creates a `CatBoost` model and returns its name."""
    name = "test"

    mlflow.catboost.log_model(
        model,
        artifact_path="model",
        registered_model_name=name,
    )
    return name


class FakeFlavor:  # pylint: disable=too-few-public-methods
    """Fake mlflow flavor module"""

    def load_model(self, *_, **__):
        """Fake load_model method."""
        return 5


def test_mlflow_model_loader_dataset(model_name: str):
    """Tests if the dataset loads the model correctly."""
    dataset = MlflowModelLoaderDataSet(
        model=model_name,
        flavor="mlflow.catboost",
    )
    model: MlflowModelLoaderDataSet = dataset.load()
    assert isinstance(model.model, CatBoostClassifier)


def test_mlflow_model_loader_model_not_found(setup_mlflow):
    """Tests if the dataset raises an error if the model is not found."""
    dataset = MlflowModelLoaderDataSet(
        model="not_found", flavor="mlflow.catboost", retry={"enabled": False}
    )
    with pytest.raises(DataSetError):
        dataset.load()


def test_mlflow_model_loader_model_stage_not_found(model_name: str):
    """Tests if the dataset errors if the model of the given stage is not found."""
    dataset = MlflowModelLoaderDataSet(
        model=model_name,
        flavor="mlflow.catboost",
        stage="not_found",
        retry={"enabled": False},
    )
    with pytest.raises(DataSetError):
        dataset.load()


def test_mlflow_model_loader_retry(model_name: str, mocker: MockFixture):
    """Tests if the dataset retries if the model is not found."""
    dataset = MlflowModelLoaderDataSet(
        model=model_name,
        flavor="mlflow.catboost",
        retry={"enabled": True, "interval": 0.05},
    )
    mock = mocker.patch.object(
        MlflowModelLoaderDataSet,
        "_mlflow_module",
        new_callable=mocker.PropertyMock,
        side_effect=[
            mlflow.MlflowException(""),
            mlflow.MlflowException(""),
            FakeFlavor(),
        ],
    )
    dataset.load()
    assert mock.call_count == 3


def test_mlflow_model_loader_retry_max_retries(model_name: str, mocker: MockFixture):
    """Tests if the dataset retries stops after the max retries."""
    dataset = MlflowModelLoaderDataSet(
        model=model_name,
        flavor="mlflow.catboost",
        retry={"enabled": True, "interval": 0.05, "max": 2},
    )
    mock = mocker.patch.object(
        MlflowModelLoaderDataSet,
        "_mlflow_module",
        new_callable=mocker.PropertyMock,
        side_effect=[
            mlflow.MlflowException(""),
            mlflow.MlflowException(""),
            FakeFlavor(),
        ],
    )
    with pytest.raises(DataSetError):
        dataset.load()
    assert mock.call_count == 2


def test_mlflow_model_loader_update(
    model_name: str, mocker: MockFixture, model: CatBoostClassifier
):
    """Tests if the dataset updates the model correctly."""
    dataset = MlflowModelLoaderDataSet(
        model=model_name,
        flavor="mlflow.catboost",
        update_interval=0.01,
    )
    mock = mocker.patch.object(dataset, "_load")
    dataset.model
    mock.assert_called()
    mlflow.catboost.log_model(
        model, artifact_path="model", registered_model_name=model_name
    )
    time.sleep(0.02)
    dataset.model
    assert mock.call_count == 2


def test_mlflow_model_loader_update_load_cached(
    model_name: str, mocker: MockFixture, model: CatBoostClassifier
):
    """Tests if the dataset caches the model before the update timer end."""
    dataset = MlflowModelLoaderDataSet(
        model=model_name,
        flavor="mlflow.catboost",
        update_interval=30.0,
    )
    mock = mocker.patch.object(dataset, "_load")
    dataset.model
    mock.assert_called()
    mlflow.catboost.log_model(
        model, artifact_path="model", registered_model_name=model_name
    )
    dataset.model
    assert mock.call_count == 1


def test_mlflow_model_loader_from_stage(setup_mlflow, model: CatBoostClassifier):
    """Tests if the dataset loads the model from the given stage correctly."""
    mlflow.catboost.log_model(
        model, artifact_path="model", registered_model_name="test"
    )
    mlflow.MlflowClient().transition_model_version_stage("test", "1", "production")
    dataset = MlflowModelLoaderDataSet(
        model="test",
        flavor="mlflow.catboost",
        stage="production",
    )
    assert isinstance(dataset.model, CatBoostClassifier)


def test_mlflow_model_loader_checks_update_of_no_model(setup_mlflow):
    """Tests if the dataset does not check for updates if there is no model."""
    dataset = MlflowModelLoaderDataSet(
        model="test",
        flavor="mlflow.catboost",
        update_interval=0.01,
        retry={"enabled": False},
    )
    with pytest.raises(DataSetError):
        dataset.model
