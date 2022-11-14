"""Tests for the scoring pipeline."""
# pylint: disable=redefined-outer-name
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from kedro.config import TemplatedConfigLoader
from pytest_mock import MockFixture

from src.hotelbookingcancellation.pipelines.scoring import create_pipeline, nodes


class FakeModel:  # pylint: disable=too-few-public-methods
    """Fake model for testing."""

    def predict(self, x: Any):
        """Fake predict method."""
        return np.zeros(len(x))


class FakeMlflowLoaderDataSet:  # pylint: disable=too-few-public-methods
    """Fake dataset for mlflow loader."""

    def __init__(self, *_, **__):
        """Init."""
        self._model = FakeModel()

    @property
    def model(self):
        """Loads the model."""
        return self._model


@pytest.fixture()
def parameters():
    """Fixture for the project parameters."""
    return TemplatedConfigLoader("./conf").get("parameters/*")


@pytest.fixture()
def client(mocker: MockFixture, parameters: dict):
    """Fixture for the API client."""
    clients = []

    def create_client(app, **_):
        client = TestClient(app)
        clients.append(client)

    mocker.patch("uvicorn.run", wraps=create_client)
    nodes.scoring_server(
        FakeMlflowLoaderDataSet(), parameters["preprocessing"], parameters["scoring"]
    )
    return clients[0]


@pytest.fixture()
def example():
    """Returns an example of a valid input."""
    return nodes.Booking.Config.schema_extra["example"].copy()


def test_scoring_server(client: TestClient, example: dict):
    """Tests if the scoring server can handle valid data."""
    res = client.post("/", json=[example])
    assert res.status_code == 200
    assert res.json() == [0]


def test_scoring_server_invalid(client: TestClient, example: dict):
    """Tests if the scoring server can handle invalid data."""
    del example["hotel"]
    res = client.post("/", json=[example])
    assert res.status_code == 422
    assert res.json() == {
        "detail": [
            {
                "loc": ["body", 0, "hotel"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
    }


def test_scoring_server_invalid_type(client: TestClient, example: dict):
    """Tests if the scoring server can handle invalid type."""
    example["lead_time"] = "wrong"
    res = client.post("/", json=[example])
    assert res.status_code == 422
    assert res.json() == {
        "detail": [
            {
                "loc": ["body", 0, "lead_time"],
                "msg": "value is not a valid integer",
                "type": "type_error.integer",
            }
        ]
    }


def test_scoring_server_invalid_date(client: TestClient, example: dict):
    """Tests if the scoring server can handle invalid date."""
    example["reservation_status_date"] = "wrong"
    res = client.post("/", json=[example])
    assert res.status_code == 422
    assert res.json() == {
        "detail": [
            {
                "loc": ["body", 0, "reservation_status_date"],
                "msg": "invalid date format",
                "type": "value_error.date",
            }
        ]
    }


def test_scoring_server_multiple(client: TestClient, example: dict):
    """Tests if the scoring server can handle multiple elements."""
    res = client.post("/", json=[example, example])
    assert res.status_code == 200
    assert res.json() == [0, 0]


def test_validate_pipeline_create():
    """Tests if a pipeline can be instantiated."""
    pipeline = create_pipeline()
    assert pipeline
