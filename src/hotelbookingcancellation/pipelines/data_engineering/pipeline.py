"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import node1


def create_pipeline() -> Pipeline:
    """Creates the pipeline for data engineering."""
    return pipeline(
        [
            node(
                func=node1,
                inputs=None,
                outputs="io1",
            )
        ]
    )
