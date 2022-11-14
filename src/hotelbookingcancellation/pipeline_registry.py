"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import data_engineering, data_science, scoring


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipelines = {
        "de": data_engineering.create_pipeline(),
        "ds": data_science.create_pipeline(),
        "scoring": scoring.create_pipeline(),
    }
    pipelines["__default__"] = pipelines["de"] + pipelines["ds"]
    return pipelines
