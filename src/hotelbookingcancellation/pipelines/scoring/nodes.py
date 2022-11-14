"""Contains the nodes for the scoring pipeline."""
from datetime import date
from typing import List, TypedDict, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ..data_engineering.nodes import _PreprocessBookingsParams, preprocess_bookings
from .mlflow_model_loader_dataset import MlflowModelLoaderDataSet


class Booking(BaseModel):
    """Booking data model for API."""

    hotel: str
    meal: str
    market_segment: str
    distribution_channel: str
    reserved_room_type: str
    deposit_type: str
    customer_type: str
    reservation_status_date: date
    lead_time: int
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    agent: int
    company: int
    adr: Union[float, int]
    required_car_parking_spaces: int
    total_of_special_requests: int

    class Config:  # pylint: disable=missing-class-docstring,too-few-public-methods
        schema_extra = {
            "example": {
                "hotel": "Resort Hotel",
                "meal": "BB",
                "market_segment": "Direct",
                "distribution_channel": "Direct",
                "reserved_room_type": "A",
                "deposit_type": "No Deposit",
                "customer_type": "Transient",
                "reservation_status_date": "2015-07-01",
                "lead_time": 342,
                "arrival_date_week_number": 27,
                "arrival_date_day_of_month": 1,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 0,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "agent": 0,
                "company": 0,
                "adr": 0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
            }
        }


class _ScoringParams(TypedDict):
    uvicorn: dict
    """Uvicorn parameters."""
    fastapi: dict
    """FastAPI parameters."""


def scoring_server(
    dataset: MlflowModelLoaderDataSet,
    preprocess_params: _PreprocessBookingsParams,
    scoring_params: _ScoringParams,
):
    """Creates a FastAPI server for scoring the model.

    Args:
        dataset: MlflowModelLoaderDataSet instance.
        preprocess_params: Preprocessing parameters.
        scoring_params: Scoring parameters.
    """
    app = FastAPI(**scoring_params.get("fastapi", {}))

    @app.post("/")
    def score(bookings: List[Booking]):
        df = pd.json_normalize([booking.dict() for booking in bookings])
        df[preprocess_params["target"]] = 0
        df = preprocess_bookings(df, preprocess_params).drop(
            columns=preprocess_params["target"]
        )
        return dataset.model.predict(df).tolist()

    uvicorn.run(app, **scoring_params.get("uvicorn", {}))
