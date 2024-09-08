FROM python:3.10.12-alpine

WORKDIR /app

ADD ./docker/dataset_builder.py ./

USER 1001