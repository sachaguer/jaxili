FROM continuumio/miniconda3

LABEL Description="JaxILI Docker Image with Python 3.12"
WORKDIR /home
ENV SHELL /bin/bash

RUN apt-get update
RUN apt-get install build-essential -y

COPY * .

RUN conda create -n jaxili python=3.12
RUN conda activate jaxili
RUN python -m pip install --upgrade pip
RUN pip install jaxili