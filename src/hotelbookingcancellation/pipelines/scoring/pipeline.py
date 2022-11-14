"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import scoring_server


def create_pipeline() -> Pipeline:
    """Creates the pipeline for scoring."""
    return pipeline(
        [
            node(
                func=scoring_server,
                inputs=["api_model", "params:preprocessing", "params:scoring"],
                outputs=None,
                name="scoring",
            )
        ]
    )
