FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r /workspace/requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace