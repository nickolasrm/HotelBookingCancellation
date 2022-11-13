"""Creates the data science pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train_test


def create_pipeline() -> Pipeline:
    """Creates the pipeline for data science."""
    return pipeline(
        [
            node(
                func=split_train_test,
                inputs=["preprocessed_hotel_bookings", "params:split_train_test"],
                outputs=["x_train", "x_test", "y_train", "y_test"],
                name="split_train_test",
            )
        ]
    )
