# HotelBookingCancellation

## Description

### Overview

This project aims to predict which customers are probably going to cancel a hotel booking given the booking data and the customer history.

### Motivation

The motivation behind this project is to avoid the loss of revenue due to cancellations. There's a lot of bureaucracy and cost in booking and 'unbooking' a room, so it's useful to know if a customer is going to cancel it or not.

### Dataflow Diagram

![Dataflow Diagram](/dataflow.png)

### Pipelines

* `de`: Performs the preprocessing over the raw data.
* `ds`: Performs the training and evaluation of the model.
* `__default__`: This pipeline is the combination of both `de` and `ds` pipelines.
* `scoring`: Starts an inference server with the trained model.

> Note: Every model and metrics outputs are saved in the `mlflow` server.

### Inputs/Features

| Artifact        | Type    | Notes            |
| --------------- | ------- | ---------------- |
| hotel_bookings  | Tabular | Fetched from web |


### Outputs

| Artifact   | Type    | Notes            |
| --------   | ------- | ---------------- |
| model      | Model   | [link](/conf/base/catalog.yml)                 |
| metrics    | Metrics | [link](/conf/base/parameters/data_science.yml) |

### Algorithm explanation

#### Data preparation

The raw data is fetched from the web and loaded into memory. Some columns are then dropped, and the rest are converted to the correct data type given the dataset description specified in [Hotel bookings dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191#!%29). Then the data is normalized and nan values are filled. This first part is parametrized and configurable through this [file](/conf/base/parameters/data_engineering.yml).

After that, the data is split into train and test sets. This is also configurable through this [file](/conf/base/parameters/data_science.yml).

#### Feature Engineering

No feature engineering was deeply performed, but some of the columns dropped in the data preparation step were dropped because of their low correlation with the target.

#### Optimization

A `CatBoostClassifier` model is generated with the parameters specified [here](/conf/base/parameters/data_science.yml) and saved into the `mlflow` server. This algorithm was chosen because of the high scores it achieved, and because its categorical features handling is suitable for the kind of data we have.

#### Evaluation

After training, the metrics are logged in the `mlflow` server. The metrics are specified in this [file](/conf/base/parameters/data_science.yml).

## Usage

This project was built using a microservice architecture, so the model is deployed in a docker container and an API is provided to interact with it.

### Installation

1. Be sure you have `docker`, and `docker-compose` installed.
2. Run `docker-compose build` to build the images.
3. Run `docker-compose up -d` to start the containers.
4. Access the `mlflow` server at `localhost:5000`.
5. Access the scoring server at `localhost:8000`.

### Scoring

After the training container is finished, the API service is going to be available at `localhost:8000`. To interact with it, you can use post requests providing regular booking data as a list of dictionaries to the request body. The response will be a list of predictions, one for each booking.

The API was built on top of `FastAPI`, which means you can check the API documentation at `localhost:8000/docs` with a real usage example.

## Development

In case of any changes to the code, make sure to run `make install-dev` to have the development tools installed. it's also recommended to leave the `mlflow` container running to have the `mlflow` server available.

### Quality

`pre-commit` is automatically installed whenever you run `make install` or `make install-dev`. This tool will run the lints specified in the [.pre-commit-config.yaml](/.pre-commit-config.yaml) file before every commit. If you want to run the lints manually, you can run `make lint` after staging your changes.

Testing is also available through `pytest`. To run the tests, run `make test`.

### Local execution

This project uses [kedro](https://kedro.readthedocs.io/en/stable/) to manage the pipelines. To run them locally, you can use the `kedro` command. For example, to run the `de` pipeline, run `kedro run --pipeline de`. To run the `__default__` pipeline, run `kedro run`.

If you want to see other available commands or get help about one, run `kedro <command> --help`.

## Authors

* Nickolas da Rocha Machado; [nickolasrochamachado@gmail.com](mailto:nickolasrochamachado@gmail.com)

## References

1. [Hotel booking demand datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191#!%29)
