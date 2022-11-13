"""This is a boilerplate pipeline 'data_science' generated using Kedro 0.18.2.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import node2


def create_pipeline() -> Pipeline:
    """Creates the pipeline for data science."""
    return pipeline(
        [
            node(
                func=node2,
                inputs="preprocessed_hotel_bookings",
                outputs="io2",
            )
        ]
    )
