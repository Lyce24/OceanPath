.PHONY: sync sync-gpu lint format lock-check test test-cov test-gpu build demo-artifact results \
	bench-serving bench-data image-serve ci

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

# Per-bag read time and DataLoader throughput, HDF5 release files versus the feature store
# (needs OCEANPATH_ROOT; copies 100 slides to BENCH_WORK, ~12 GB).
BENCH_WORK ?= /tmp/oceanpath-bench
bench-data:
	uv run --no-sync python scripts/benchmark_data_loading.py \
		--manifest manifests/public/tcga_nsclc_labels.csv \
		--features-root $(OCEANPATH_ROOT)/features/UNI2_h_features \
		--work $(BENCH_WORK) --out results/data_loading

image-serve:
	docker build -f docker/serve.Dockerfile -t oceanpath-serve .

# Local mirror of the blocking CI jobs.
ci: lock-check lint test-cov build

# Rebuild the result tables and figures from finished runs and refresh the README blocks
# (needs OCEANPATH_ROOT).
results:
	uv run --no-sync python scripts/collect_supervised_results.py --out results/supervised
	uv run --no-sync python scripts/collect_label_efficiency.py --out results/label_efficiency
	uv run --no-sync python scripts/update_readme_results.py
