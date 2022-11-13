"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_bookings


def create_pipeline() -> Pipeline:
    """Creates the pipeline for preprocessing the raw data."""
    return pipeline(
        [
            node(
                func=preprocess_bookings,
                inputs=["hotel_bookings", "params:preprocessing"],
                outputs="preprocessed_hotel_bookings",
            )
        ]
    )
