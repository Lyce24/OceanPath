# OceanPath

OceanPath is a supervised multiple-instance learning foundation for computational
pathology. The `main` branch owns the reusable path from whole-slide images to a
portable model artifact. BLCA is the reference integration used to keep that path
executable without putting dataset-specific behavior in the library.

## Foundation pipeline

```text
WSIs ──> patch features ──> mmap ──┐
                                   ├──> train ──> evaluate ──> export
manifest ────────────────> splits ─┘
```

The essential stages are:

1. `extract_features` — adapt TRIDENT and produce per-slide H5 features.
2. `build_mmap` — validate and pack H5 files into chunked mmap storage.
3. `split_data` — produce leakage-safe, integrity-hashed assignments.
4. `train_model` — train supervised MIL folds and final models.
5. `evaluate` — report OOF slide/patient metrics; when a held-out test cohort
   exists, also report the preselected final model without reusing OOF
   predictions or selecting a model on the test set.
6. `export_model` — create validated ONNX/TorchScript artifacts and a model card.

Split generation depends only on the manifest. It can run independently of feature
extraction and mmap construction; training waits for both branches.

## Setup

Python 3.10 through 3.12 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --group extract
export OCEANPATH_ROOT=/absolute/path/to/oceanpath-data
```

Several pathology encoders require accepting the model owner's Hugging Face
license or access terms. Request access on the model page before extraction,
then authenticate with `hf auth login` or set `HF_TOKEN`; installing the
`extract` dependency group does not grant model access.

The root contains machine data, not source code:

```text
$OCEANPATH_ROOT/
├── slides/blca/
├── manifests/blca.csv
├── features/
├── mmap/
├── splits/
└── outputs/
```

See `examples/blca/manifest.example.csv` for the reference manifest columns.

The operational encoder profiles shipped by `main` are UNI v1, CONCH v1/v1.5,
ResNet-50, Virchow v1/v2, Prov-GigaPath, and H-Optimus-1. Each
`configs/encoder/*.yaml` profile owns its feature dimension, model source,
adapter, magnification, patch size, and output subdirectory. Gemma 4 E4B/26B
profiles preserve model metadata for future work, but are explicitly
experimental: extraction fails validation until a tested Hugging Face adapter
defines image preprocessing and soft-token pooling.
For immutable production provenance, set `encoder.checkpoint_path` to a pinned
local checkpoint; OceanPath hashes checkpoint bytes into the extraction
manifest and pipeline fingerprint.

The default environment is CPU-testable. Optional installs are explicit:

```bash
uv sync --group extract  # TRIDENT and WSI readers (repository-only Git sources)
uv sync --extra serve    # FastAPI serving
uv sync --extra track    # optional W&B experiment tracking
```

The repository-only extraction group pins CONCH, OpenSDPC, and TRIDENT to
reviewed Git commits; update those revisions and `uv.lock` together.

## Run BLCA

Inspect the complete DAG without running compute:

```bash
uv run python scripts/pipeline.py pipeline.dry_run=true
```

Run individual stages:

```bash
uv run python scripts/extract_features.py
uv run python scripts/build_mmap.py
uv run python scripts/make_splits.py
uv run python scripts/train.py
uv run python scripts/evaluate.py
uv run python scripts/export_model.py
```

Hydra overrides compose platform, dataset, encoder, split, model, and training
profiles:

```bash
uv run python scripts/train.py model=transmil training.lr=3e-4
uv run python scripts/make_splits.py splits=blca_custom dry_run=true
```

Split profile names follow fold/repeat and seed overrides. Persisted split
assignments also record a config-plus-manifest identity and are never reused
under the same directory when that identity differs; use a new `splits.name`,
or `force=true` when replacement is intentional.

Training uses the same fail-closed rule. Material model, training, platform, or
split-config changes automatically receive a semantic fingerprint suffix in
`train_dir`. In-place manifest, split, mmap, or preload-checkpoint changes still
cannot resume an incompatible run. Completed folds and runs carry hashed config,
prediction, metric, and checkpoint evidence rather than relying on file
existence alone. Set `train_dir` explicitly only to resume or migrate a
deliberately named run; identity mismatches then fail closed.

The foundation finalizes and exports `best_fold`, chosen from validation
evidence before held-out evaluation. Held-out metrics are reporting-only. A
project that needs ensemble or refit artifacts must opt into those strategies
explicitly and define any model-selection cohort separately from the final test.

Use `--cfg job --resolve` on any script to inspect its resolved contract before
running it.

External experiment tracking is opt-in. Set `wandb.enabled=true` for W&B, and
also set `wandb.offline=true` when runs must remain local.
Before each stage, the default runtime performs a frozen, environment-aware uv
audit; set `runtime.strict_environment=true` to turn a mismatch warning into a
hard failure.

## Source layout

```text
src/oceanpath/
├── config/       # cfg access and the single filesystem path authority
├── contracts/    # stable stage names and serializable workflow results
├── extraction/   # typed TRIDENT adapter; no Hydra or CLI
├── storage/      # typed mmap build/read validation (no plotting)
├── splitting/    # manifest validation and split generation
├── datasets/     # training-time mmap Dataset and Lightning DataModule
├── models/       # MIL aggregators and classifier construction
├── training/     # fold planning, the Lightning module, and callbacks
├── eval/         # inference, metrics, and evaluation-report rendering
├── export/       # portable artifacts and model cards
├── serving/      # optional HTTP serving of an exported artifact
├── viz/          # storage/data diagnostic rendering such as coverage QC
├── workflows/    # the only library layer allowed to translate DictConfig
├── runtime/      # logging, provenance, run status, experiment-tracker seam
└── pipeline/     # dataset-neutral DAG, freshness checks, and completion records
```

Training mechanics that operate on domain values (the Lightning module and
callbacks) live in `training/`; the config-aware assembly of folds, identity
evidence, and post-CV finalization lives in `workflows/`. Experiment tracking is
constructed only in `runtime/reporting.py` — no workflow imports `wandb`
directly.

Top-level `scripts/` contains only 15-line Hydra launchers. Standalone storage
diagnostics live in `tools/` and are not pipeline stages.

The dependency direction is:

```text
scripts -> workflows -> domain packages
                    \-> runtime/config/contracts
pipeline -> config/contracts + domain/workflow validators
```

Domain packages must not import Hydra, parse CLI arguments, or decide experiment
paths. Scripts contain no model logic, metric math, or dataset rules; architecture
tests enforce both boundaries.

## Add a project branch

A project branch starts from `main` and adds only its delta:

- Dataset paths, column mappings, and task class metadata go in
  `configs/data/<project>.yaml`.
- Standard split parameter changes go in `configs/splits/<project>.yaml`.
- Training hyperparameters go in `configs/training/<project>.yaml`.
- Experiment naming and train/eval output layout go in
  `configs/experiment/<project>.yaml`.
- A genuinely new split algorithm goes in `oceanpath/splitting/`, with a typed
  contract and tests; it is not an `if project == ...` condition in core code.
- New model architectures go in `oceanpath/models/` with a model config and model
  contract tests.
- Project-only analysis belongs in a project package or project scripts and is not
  registered in the foundation DAG.
- Swapping the experiment tracker (W&B → MLflow, TensorBoard, or none) is a single
  edit in `oceanpath/runtime/reporting.py`; workflows are tracker-agnostic.
- A pretraining project may add its own package, configs, workflows, and DAG
  factory while reusing extraction, storage, datasets, runtime, and contracts.

A downstream cohort can therefore replace split, training, or evaluation
profiles and implementations while retaining the same stage inputs, outputs,
path layout, and run metadata.

## Verification

```bash
make ci
```

`make ci` runs file-hygiene hooks, verifies the lockfile, applies one Ruff policy
to the whole repository, checks stable contracts, shared utilities, and model
interfaces, runs the CPU unit suite, and builds the source distribution and
wheel. Every essential root config is composed in
`tests/test_foundation.py`. Pipeline fingerprints include domain source
directories, so changing model or workflow code invalidates stale outputs.
The blocking mypy gate covers the stable backbone surface; it does not claim
that every research workflow is fully typed.
The wheel contains the reusable `oceanpath` library; Hydra configs and thin
launchers remain repository-first so project branches can own their composition.
