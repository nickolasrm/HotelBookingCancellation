FROM python:3.9.15-slim-buster

ENV HOME /home

# Mlflow settings
ENV MLFLOW_REGISTRY_URI "sqlite://${HOME}/mlruns.db"
ENV MLFLOW_TRACKING_URI "file://${HOME}/mlruns"
ENV MLFLOW_HOST "0.0.0.0"

# Preparing mlflow
RUN pip install mlflow
RUN mkdir -p ${HOME}/mlruns

CMD mlflow server \
    --backend-store-uri ${MLFLOW_TRACKING_URI} \
    --registry-store-uri ${MLFLOW_REGISTRY_URI} \
    --host ${MLFLOW_HOST}
