# syntax=docker/dockerfile:1
#
# CPU inference image: FastAPI + ONNX Runtime serving one exported artifact
# directory (scripts/export_model.py writes model.onnx + model_card.json), mounted
# at /models. There is no torch in the image: it installs only the serve-runtime
# dependency group from uv.lock, plus the oceanpath package itself.
#
#   docker build -f docker/serve.Dockerfile -t oceanpath-serve .
#   docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts/demo:/models:ro" oceanpath-serve
#   curl -s http://127.0.0.1:8000/health

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-default-groups --only-group serve-runtime
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python --no-deps .

FROM python:${PYTHON_VERSION}-slim
RUN useradd --create-home --uid 1000 oceanpath
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OCEANPATH_ARTIFACT_DIR=/models
USER oceanpath
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
ENTRYPOINT ["python", "-m", "oceanpath.serving"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
