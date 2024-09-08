FROM python:3.10.12-alpine

WORKDIR /app

ADD ./docker/main.py ./

RUN pip install roboflow ultralytics

USER 1001