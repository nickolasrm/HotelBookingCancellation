FROM python:3.9.15-slim-buster

ENV HOME /home
ENV PROJECT ${HOME}/app

# Project settings
ENV MLFLOW_TRACKING_URI "http://localhost:5000"

# Preparing working directory
RUN mkdir ${PROJECT}
WORKDIR ${PROJECT}

# Installing system dependencies
RUN apt-get update
RUN apt-get install -y make

# Copying kedro project
RUN mkdir -p conf/base
COPY ./conf/base ./conf/base
COPY ./src ./src
COPY ./Makefile .
COPY ./pyproject.toml .

# Building project
RUN make install
RUN kedro package
RUN pip install dist/*.whl
RUN rm -r src Makefile dist
RUN mkdir conf/local logs
