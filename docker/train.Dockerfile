# syntax=docker/dockerfile:1
#
# GPU image for extraction, pretraining, supervised training and linear probing.
# torch's CUDA wheels bundle the CUDA runtime, so the default slim base runs on
# any host with the NVIDIA container toolkit. The Mamba-2 aggregators compile
# CUDA kernels and need nvcc, so build them on a CUDA devel base:
#
#   docker build -f docker/train.Dockerfile -t oceanpath-train .
#   docker build -f docker/train.Dockerfile -t oceanpath-train:mamba \
#       --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04 --build-arg WITH_MAMBA=1 .
#
#   docker run --rm --gpus all --shm-size 16g -v /path/to/oceanpath-data:/data \
#       oceanpath-train scripts/train_supervised.py data=tcga_nsclc encoder=uni2h model=abmil

ARG BASE_IMAGE=debian:bookworm-slim
FROM ${BASE_IMAGE}
ARG WITH_MAMBA=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    MAMBA_FORCE_BUILD=TRUE \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/oceanpath
COPY pyproject.toml uv.lock README.md LICENSE .python-version ./
RUN extras="--extra extract"; \
    if [ "$WITH_MAMBA" = "1" ]; then extras="$extras --extra mamba"; fi; \
    uv sync --locked --no-dev --no-install-project $extras
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN extras="--extra extract"; \
    if [ "$WITH_MAMBA" = "1" ]; then extras="$extras --extra mamba"; fi; \
    uv sync --locked --no-dev $extras

# configs/platform/local.yaml derives every storage root from OCEANPATH_ROOT.
ENV OCEANPATH_ROOT=/data
VOLUME ["/data"]
ENTRYPOINT ["python"]
CMD ["scripts/pipeline.py", "--help"]
