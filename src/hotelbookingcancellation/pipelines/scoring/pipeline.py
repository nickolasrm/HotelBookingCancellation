"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import node3


def create_pipeline() -> Pipeline:
    """Creates the pipeline for scoring."""
    return pipeline(
        [
            node(
                func=node3,
                inputs=None,
                outputs="io3",
            )
        ]
    )
