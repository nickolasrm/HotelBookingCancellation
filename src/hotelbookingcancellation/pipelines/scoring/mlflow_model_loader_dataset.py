"""DataSet for loading mlflow models from registry."""
import importlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import mlflow  # type: ignore
import mlflow.exceptions  # type: ignore
from kedro.io import AbstractDataSet, DataSetError


class MlflowLoaderFlavor(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for Mlflow flavors loaders."""

    def load_model(self, path: str, **kwargs):
        """Load model from path."""


@dataclass
class _Update:
    enabled: bool = True
    max: int = -1
    interval: float = 0.5
    last: float = float("-inf")
    data: Any = None


class MlflowModelLoaderDataSet(AbstractDataSet):
    """Continuously loads a model from the `Model Registry`."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        flavor: str,
        stage: Optional[str] = None,
        retry: Optional[Dict[str, Any]] = None,
        update_interval: Optional[float] = None,
    ):
        """Initializes the dataset.

        Args:
            model (str): The name of the model in the Model Registry.
            flavor (str): The flavor of the model.
            stage (Optional[str]): Stage or version of the model to load.
                Defaults to 'latest'.
            retry (Optional[Dict[str, Any]]): Whether to retry the load if it fails.
                Defaults to None.
            update_interval (Optional[float]): Interval between checks for
                updated model in seconds. Defaults to 5.
        """
        self._model_name = model
        self._flavor = flavor
        self._stage = stage or "latest"
        self._retry = _Update(**retry) if retry else _Update()
        self._update = _Update(interval=update_interval or 10.0)
        self._model = _Update()

    @property
    def _mlflow_module(self) -> MlflowLoaderFlavor:
        """Loads the mlflow flavor module.

        Returns:
            MlflowLoaderFlavor: The mlflow flavor module.
        """
        return importlib.import_module(self._flavor)

    @property
    def _model_uri(self) -> str:
        """Gets the model uri.

        Returns:
            str: The model uri.
        """
        return f"models:/{self._model_name}/{self._stage}"

    def _check_updated(self) -> bool:
        """Fetches the timestamp of the model.

        Returns:
            bool: Whether the model has been updated.
        """
        update_time = time.perf_counter()
        if update_time > (self._update.last + self._update.interval):
            self._update.last = update_time
            model_timestamp = (
                mlflow.MlflowClient()
                .get_registered_model(self._model_name)
                .last_updated_timestamp
            )
            if model_timestamp != self._model.last:
                self._model.last = model_timestamp
                return False
        return True

    @property
    def model(self) -> Any:
        """Gets the current model."""
        try:
            if not self._check_updated():
                self._load()
        except mlflow.MlflowException:
            self._load()
        return self._model.data

    def _load(self) -> Any:
        """Loads the model.

        Returns:
            Any: The loaded model.

        Raises:
            mlflow.MlflowException: If the model is not found after retries.
        """
        retry = 0
        max_retries = self._retry.max if self._retry.enabled else 1
        while retry != max_retries:
            try:
                self._model.data = self._mlflow_module.load_model(self._model_uri)
                return self
            except mlflow.MlflowException:
                self._logger.warning(
                    "Failed to load model '%s' from stage '%s'",
                    self._model_name,
                    self._stage,
                )
                retry += 1
                time.sleep(self._retry.interval)
        raise DataSetError(
            f"Failed to load model '{self._model_name}' from stage '{self._stage}'"
        )

    def _save(self, _: Any):
        """Saves the model."""
        raise NotImplementedError(
            "Saving is not implemented for MlflowModelLoaderDataSet"
        )

    def _describe(self) -> dict:
        """Describes the dataset.

        Returns:
            dict: The dataset description.
        """
        return dict(
            model=self._model,
            flavor=self._flavor,
            stage=self._stage,
            retry=self._retry,
            update=self._update,
        )
