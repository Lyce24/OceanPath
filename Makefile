.PHONY: sync sync-gpu lint format lock-check test test-cov test-gpu build demo-artifact \
	bench-serving image-serve ci

# CPU environment: everything except the Mamba-2 CUDA kernels.
sync:
	uv sync --locked --extra serve

# GPU environment: compiles mamba-ssm and causal-conv1d (needs nvcc; ~30 min once).
sync-gpu:
	MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE uv sync --locked --extra serve --extra mamba

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

format:
	uv run --no-sync ruff check --fix .
	uv run --no-sync ruff format .

lock-check:
	uv lock --check

test:
	uv run --no-sync pytest -m "not slow and not gpu and not integration"

test-cov:
	uv run --no-sync pytest -m "not slow and not gpu and not integration" \
		--cov --cov-report=xml --cov-report=term

# CUDA and Mamba tests; run on a GPU machine after `make sync-gpu`.
test-gpu:
	uv run --no-sync pytest -m gpu

build:
	uv build

# Random-weight ABMIL exported to ONNX + TorchScript, for serving smoke tests.
demo-artifact:
	uv run --no-sync python scripts/make_demo_artifact.py --out artifacts/demo

# p50/p95 latency per slide for PyTorch, TorchScript and ONNX across bag sizes.
bench-serving:
	uv run --no-sync python scripts/benchmark_serving.py --out results/serving_latency

image-serve:
	docker build -f docker/serve.Dockerfile -t oceanpath-serve .

# Local mirror of the blocking CI jobs.
ci: lock-check lint test-cov build
