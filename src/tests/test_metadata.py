"""Tests if duplicated metadata matches across multiple parts of the code."""
import pytest
import tomli

from src.hotelbookingcancellation import __version__


@pytest.fixture(name="metadata")
def fixture_metadata() -> dict:
    """Returns metadata from pyproject.toml."""
    with open("pyproject.toml", "rb") as file:
        toml = tomli.load(file)
        return toml["project"]


def test_version(metadata: dict):
    """Checks if version matches."""
    assert __version__ == metadata["version"]
