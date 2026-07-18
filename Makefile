.PHONY: sync config lint precommit typecheck lock-check test test-cov build ci

UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
UV_TOOL_DIR ?= $(CURDIR)/.uv-tools
PRE_COMMIT_HOME ?= $(CURDIR)/.cache/pre-commit
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
export UV_CACHE_DIR UV_TOOL_DIR PRE_COMMIT_HOME MPLCONFIGDIR

sync:
	uv sync --frozen

config:
	uv run pytest -q tests/test_foundation.py

# ── Static quality ────────────────────────────────────────────────────────────
lint:
	uv run ruff check .
	uv run ruff format --check .

# File hygiene + YAML lint + import order via the pinned pre-commit hooks.
# Uses uvx so it never needs the (heavy) project environment.
precommit:
	uvx --from pre-commit==4.5.1 pre-commit run --all-files --show-diff-on-failure

# Fails if uv.lock is out of date with pyproject.toml.
lock-check:
	uv lock --check

# Stable public contracts, shared utilities, and model interfaces are blocking.
# Broaden this list as workflow internals are typed; do not add an advisory job
# that is allowed to stay red.
typecheck:
	uv run mypy \
		src/oceanpath/config/access.py \
		src/oceanpath/config/paths.py \
		src/oceanpath/contracts \
		src/oceanpath/extraction \
		src/oceanpath/models \
		src/oceanpath/pipeline \
		src/oceanpath/runtime/logging.py \
		src/oceanpath/splitting \
		src/oceanpath/storage \
		src/oceanpath/training/folds.py \
		src/oceanpath/utils/json.py \
		src/oceanpath/utils/repro.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	uv run pytest -m "not slow and not gpu and not integration"

test-cov:
	uv run pytest -m "not slow and not gpu and not integration" \
		--cov --cov-report=xml --cov-report=term-missing

# Verify that the reusable library surface is packageable. Repository Hydra
# configs and launchers intentionally remain repo-first rather than wheel CLIs.
build:
	uv build

# Local one-shot gate (mirrors every blocking CI job).
ci: precommit lock-check lint typecheck test-cov build
