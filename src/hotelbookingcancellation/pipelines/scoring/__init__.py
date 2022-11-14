"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.18.2
"""

from .mlflow_model_loader_dataset import MlflowModelLoaderDataSet
from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
