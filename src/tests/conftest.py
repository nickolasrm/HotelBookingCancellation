"""Common fixtures for testing."""
from pathlib import Path

import mlflow
import pytest


@pytest.fixture()
def setup_mlflow(tmp_path: Path):
    """Sets up mlflow."""
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns.as_posix()}")
    mlflow.set_registry_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    exp = mlflow.create_experiment("test")
    mlflow.start_run(experiment_id=exp)


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Removes all mlflow runs."""
    yield
    while mlflow.active_run():
        mlflow.end_run()
