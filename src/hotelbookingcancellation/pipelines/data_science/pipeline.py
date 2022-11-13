"""Creates the data science pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate, optimize, split_train_test


def create_pipeline() -> Pipeline:
    """Creates the pipeline for data science."""
    return pipeline(
        [
            node(
                func=split_train_test,
                inputs=["preprocessed_hotel_bookings", "params:split_train_test"],
                outputs=["x_train", "x_test", "y_train", "y_test"],
                name="split_train_test",
            ),
            node(
                func=optimize,
                inputs=["x_train", "y_train", "params:optimize"],
                outputs="model",
                name="optimize",
            ),
            node(
                func=evaluate,
                inputs=["model", "x_test", "y_test", "params:evaluate"],
                outputs="metrics",
                name="evaluate",
            ),
        ]
    )
