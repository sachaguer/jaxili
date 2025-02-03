FROM continuumio/miniconda3

LABEL Description="JaxILI Docker Image with Python 3.12"
WORKDIR /home
ENV SHELL /bin/bash

ENV ENV_NAME=jaxili

RUN apt-get update
RUN apt-get install build-essential -y

RUN conda create -n $ENV_NAME python=3.12 -y && \
    conda init bash

RUN /bin/bash -c "source activate $ENV_NAME && pip install jaxili"

ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

CMD ["bash"]