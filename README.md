# OceanPath

MIL pretraining, benchmarking, and deployment for computational pathology.

OceanPath provides a DAG-orchestrated pipeline that takes whole-slide images from feature extraction through self-supervised pretraining or supervised training, evaluation, statistical analysis, and model export — all configured via Hydra and executed with make-like freshness checks.

## Overview

Two pipeline modes cover the main workflows:

**Supervised** (7 stages) — extract features, build mmap store, split data, train a MIL classifier with k-fold CV, evaluate, analyze, export.

**Pretraining** (4 stages) — build mmap store, split train/val, SSL pretrain an aggregator (VICReg / SimCLR / BYOL / DINO / JEPA), export.

Both modes share the same DAG engine, atomic transaction system, and fingerprint-aware caching so that changing a config key automatically invalidates downstream stages.

## Repo Structure

```
OceanPath/
├── configs/                    # Hydra config groups
│   ├── pipeline.yaml           #   root config for scripts/pipeline.py
│   ├── train.yaml              #   root config for scripts/train.py
│   ├── pretrain.yaml           #   root config for scripts/pretrain.py
│   ├── pipeline_profile/       #   supervised.yaml, pretrain_only.yaml
│   ├── data/                   #   dataset metadata (csv path, slide dir, mmap dir)
│   ├── encoder/                #   feature extractor (UNI v1/v2, UNI-2H)
│   ├── model/                  #   MIL architecture (abmil, transmil, perceiver, …)
│   ├── training/               #   supervised hyperparameters
│   ├── pretrain_training/      #   SSL method + augmentation config
│   ├── splits/                 #   CV scheme (kfold5, holdout, nested, mccv)
│   ├── storage/                #   mmap chunking settings
│   ├── extraction/             #   TRIDENT extraction settings
│   └── platform/               #   paths, compute environment
├── scripts/                    # Entry points (one per stage + pipeline runner)
│   ├── pipeline.py             #   DAG orchestrator
│   ├── extract_features.py     #   stage 1: WSI → H5 features
│   ├── pretraining_data_cleanining.py  # pretraining manifest curation
│   ├── build_mmap.py           #   stage 2: H5 → chunked binary store
│   ├── visualize_mmap_coverage.py      # QC: spatial coverage after capping
│   ├── make_splits.py          #   stage 3: generate fold assignments
│   ├── train.py                #   stage 4: k-fold supervised training
│   ├── pretrain.py             #   stage 3 (pretrain): SSL pretraining
│   ├── benchmark_batching.py   #   benchmark SSL batching strategies
│   ├── evaluate.py             #   stage 5: metrics + plots
│   ├── analyze.py              #   stage 6: statistical analysis
│   ├── export_model.py         #   stage 7: TorchScript + ONNX export
│   ├── compare_experiments.py  #   cross-experiment significance tests
│   ├── benchmark.py            #   benchmark model architectures
│   └── serve.py                #   FastAPI inference server
├── src/oceanpath/
│   ├── pipeline/               # DAG engine, atomic transactions
│   ├── data/                   # mmap builder, datasets, batching, data modules, splits
│   ├── models/                 # MIL aggregators + registry + WSI classifier
│   ├── ssl/                    # SSL losses, heads, augmentation, pretrain module
│   ├── modules/                # Lightning training module, callbacks, finalization
│   ├── eval/                   # metrics, plots, bootstrap CI, comparison tests
│   ├── serving/                # ONNX inference, FastAPI endpoint
│   └── utils/                  # config fingerprinting, provenance capture
└── tests/
    ├── test_models.py          # aggregator contracts, gradients, masking, serialization
    ├── test_pipeline.py        # DAG sorting, freshness, atomic transactions
    ├── test_data.py            # datasets, dataloaders
    ├── test_pretrain_data.py   # mmap-backed pretrain dataset + dual-view collation
    ├── test_ssl_pretrain.py    # SSL losses, pretrain module forward/backward
    └── test_eval.py            # metric computation, statistical tests
```

## Cross-cutting Engineering

| Concern | Mechanism |
|---|---|
| **Configuration** | Hydra config groups with composition (`data=gej encoder=univ1 model=abmil`) and `${interpolation}` |
| **Reproducibility** | Config fingerprinting — every stage output is tagged with a hash of its relevant config keys; stale fingerprints trigger re-execution |
| **Atomicity** | Stage outputs are written to a `.tmp` dir, validated, then atomically renamed; failures roll back cleanly |
| **Freshness** | Make-like mtime + fingerprint checks — stages are skipped when outputs are newer than inputs and fingerprints match |
| **AMP safety** | Float32 softmax under `autocast`, logit clamping, `nan_to_num` that preserves the computation graph |
| **Memory** | Gradient checkpointing per model, mmap-backed datasets (never load full H5 into memory), chunked binary writes |
| **Extensibility** | Registry + factory pattern for aggregators (`register_aggregator` / `build_aggregator`); new architectures are a single file + one registry line |

## Data Flow

### Supervised Pipeline

```
WSI slides (.svs/.sdpc)
  │
  ▼  extract_features ── TRIDENT encoder (UNI, etc.)
Per-slide H5 files (features [N, D] + coords [N, 2])
  │
  ▼  build_mmap ── streaming two-pass scan+write
Chunked binary store (features_*.bin, coords_*.bin, index_arrays.npz)
  │
  ▼  split_data ── k-fold / holdout / nested CV / MCCV
Fold assignments (fold_*.parquet)
  │
  ▼  train_model ── Lightning k-fold loop
Per-fold checkpoints + OOF predictions + embeddings
  │
  ▼  evaluate ── bootstrap CI, calibration, operating points
metrics.json, ROC/PR curves, histograms
  │
  ▼  analyze ── DeLong, McNemar, UMAP, attention maps
Statistical comparisons + visualizations
  │
  ▼  export_and_serve ── TorchScript + ONNX
Deployment artifacts
```

### SSL Pretraining Pipeline

```
Pre-extracted features (H5 dir, no labels required)
  │
  ▼  pretraining_data_cleanining.py ── filter sources, discard small slides
Curated manifest CSV (slide_id, cancer_type, num_patches)
  │
  ▼  build_mmap ── with cap_strategy (random / spatial_stratified)
Chunked binary store + coverage QC thumbnails
  │
  ▼  split_data ── random 90/10 or manifest-based
split_manifest.json (train_ids, val_ids)
  │
  ▼  pretrain_model ── batching strategy + augmentation → aggregator → SSL loss
Aggregator checkpoint (ready for downstream fine-tuning)
  │
  ▼  export_and_serve
Deployment artifacts
```

## Pipeline Stages

### Stage 1 — Feature Extraction

| | |
|---|---|
| **Goal** | Convert WSIs into fixed-size patch embeddings |
| **Raw inputs** | Whole-slide images (`.svs`, `.sdpc`) |
| **Entry** | `scripts/extract_features.py` |
| **Core modules** | `oceanpath.data.feature_extract`, TRIDENT |
| **Reads** | Slide directory |
| **Writes** | Per-slide `.h5` files containing `features [N, D]` and `coords [N, 2]` |
| **Techniques** | Pretrained vision encoder (UNI v1/v2, UNI-2H); patch tiling at 20x |

### Pretraining Data Cleaning

| | |
|---|---|
| **Goal** | Curate a clean manifest of slides suitable for SSL pretraining |
| **Entry** | `scripts/pretraining_data_cleanining.py` |
| **Reads** | H5 feature directory (scans shapes only, no data loaded) |
| **Writes** | `pretrain_manifest.csv` (slide_id, slide_path, cancer_type, num_patches), `pretrain_manifest_stats.txt` |
| **Techniques** | Source filtering (TCGA + CPTAC only); minimum patch count threshold (default 256); cancer-type extraction from directory structure; per-type breakdown stats |

Run this before `build_mmap` when preparing pretraining data. The output manifest can be used with `storage.filter_by_csv=true` to restrict which slides enter the mmap store.

### Stage 2 — Build MMap Store

| | |
|---|---|
| **Goal** | Consolidate per-slide H5 files into a fast, mmap-readable binary store |
| **Raw inputs** | H5 feature directory |
| **Entry** | `scripts/build_mmap.py` |
| **Core modules** | `oceanpath.data.mmap_builder` |
| **Reads** | `*.h5` files |
| **Writes** | `features_*.bin`, `coords_*.bin`, `index_arrays.npz` (slide IDs, offsets, lengths, chunk map); `coverage_qc/` directory with per-slide spatial thumbnails |
| **Techniques** | Two-pass streaming (scan shapes, then write in 4096-patch chunks); source hashing for change detection; build-time patch capping with pluggable strategies |

**Build-time capping** (`storage.max_instances`): When slides exceed the cap, patches are subsampled using one of two strategies:

| Strategy | Behavior | Best for |
|---|---|---|
| `spatial_stratified` (default) | Divides the slide into a grid, allocates slots proportional to sqrt(cell density). Preserves spatial coverage and local density variation. | SSL with coords-aware augmentations (SpatialCrop, LocalFeatureSmooth, etc.) |
| `random` | Uniform random subsample. | Simple baselines, when spatial structure is not important |

Configure in `configs/storage/mmap.yaml`:
```yaml
max_instances: 16384
cap_strategy: spatial_stratified
cap_grid_size: 32
```

**Coverage QC**: When `save_coverage_thumbnails: true`, the build writes per-slide spatial maps to `{mmap_dir}/coverage_qc/` showing kept (green) vs discarded (red) patches. A standalone visualization script is also available:

```bash
python scripts/visualize_mmap_coverage.py \
  --h5_dir /path/to/h5 --mmap_dir /path/to/mmap \
  --output_dir outputs/coverage_qc --only_capped
```

### Stage 3 — Split Data

| | |
|---|---|
| **Goal** | Assign slides to train/val/test folds |
| **Raw inputs** | CSV with slide metadata + labels |
| **Entry** | `scripts/make_splits.py` |
| **Core modules** | `oceanpath.data.splits` |
| **Reads** | CSV, mmap index |
| **Writes** | `fold_*.parquet` or `split_manifest.json` (pretraining) |
| **Techniques** | Stratified k-fold, holdout, nested CV, Monte-Carlo CV; label-stratified or random |

### Stage 4a — Supervised Training

| | |
|---|---|
| **Goal** | Train a MIL classifier with k-fold cross-validation |
| **Raw inputs** | MMap store + fold assignments |
| **Entry** | `scripts/train.py` |
| **Core modules** | `oceanpath.modules.train_module` (`MILTrainModule`), `oceanpath.models`, `oceanpath.data.datamodule` |
| **Reads** | MMap store, fold parquets, optional pretrained aggregator weights |
| **Writes** | Per-fold `best.ckpt`, `preds_val.parquet`, `embeddings.npy`; cross-fold `cv_results.json`, `oof_predictions.parquet` |
| **Techniques** | CE / BCE / Focal loss; AdamW + cosine LR; early stopping; fold-level resume; optional aggregator freezing for fine-tuning from pretrained weights |

### Stage 4b — SSL Pretraining

| | |
|---|---|
| **Goal** | Learn slide-level representations without labels |
| **Raw inputs** | MMap store + split manifest |
| **Entry** | `scripts/pretrain.py` |
| **Core modules** | `oceanpath.ssl.pretrain_module`, `oceanpath.ssl.losses`, `oceanpath.ssl.augmentation`, `oceanpath.data.batching`, `oceanpath.data.pretrain_datamodule` |
| **Reads** | MMap store, split manifest |
| **Writes** | Aggregator checkpoint, training logs |
| **Techniques** | Five SSL methods, per-method batching strategies, four-category augmentation pipeline; quality monitoring via RankMe (effective rank) and alpha-ReQ (power-law exponent) |

**SSL methods**: **VICReg** (variance-invariance-covariance), **SimCLR** (NT-Xent contrastive), **BYOL** (bootstrap with EMA), **DINO** (self-distillation with centering+sharpening), **JEPA** (joint-embedding predictive).

#### Batching Strategies

Different SSL methods benefit from different batching strategies. Nine named strategies are provided (`oceanpath.data.batching`), all producing a uniform output dict so the SSL module requires zero changes:

| Strategy | Idea | Default for |
|---|---|---|
| `pad_to_max_in_batch` | Pad all bags to the longest in the batch | General default |
| `pad_to_global` | Pad to a fixed global max length | — |
| `token_budget` | Dynamic batch size targeting a fixed token count | **JEPA** (avoids padding for variable-length context/target masks) |
| `bucket_batching` | Sort slides by length into buckets, fixed B per bucket | — |
| `subsample_fixed_n` | Subsample every slide to a fixed N patches | **VICReg**, **BYOL**, **SimCLR** (dual-view contrastive methods) |
| `regional_crops` | TITAN-style fixed-size spatial crops | — |
| `sequence_packing` | Pack multiple slides into one sequence with block-diagonal attention | — |
| `multi_crop` | DINOv2/iBOT: 2 global + N local views | **DINO** |
| `jepa` | Context/target masking partition | **JEPA** (alternative to token_budget) |

Per-method defaults are set in `configs/pretrain_training/{method}.yaml` and can be overridden:

```bash
# JEPA with token budget (default)
python scripts/pretrain.py pretrain_training=jepa

# Override: JEPA with sequence packing instead
python scripts/pretrain.py pretrain_training=jepa \
  pretrain_training.batching.strategy=sequence_packing
```

**Benchmarking batching strategies**: `scripts/benchmark_batching.py` measures epoch time, padding ratio, batch size distribution, token utilization, and peak memory across all strategies on real mmap data:

```bash
python scripts/benchmark_batching.py --mmap_dir /path/to/mmap

# Only specific strategies
python scripts/benchmark_batching.py --mmap_dir /path/to/mmap \
  --strategies token_budget subsample_fixed_n bucket_batching

# With SSL augmentation enabled (default: identity for clean batching benchmark)
python scripts/benchmark_batching.py --mmap_dir /path/to/mmap --augmentation
```

#### Augmentation Pipeline

Feature-space augmentations are organized into four categories, ordered from strongest to subtlest:

| Category | Transforms | Purpose |
|---|---|---|
| **Patch selection** (which patches each view sees) | Spatial crop, density-aware subsampling, region drop, instance dropout, random subsampling | Primary source of view diversity — different spatial coverage per view |
| **Coords-aware feature perturbation** | LocalFeatureSmooth (k-NN blending), SpatialFeatureInterpolation (lerp toward neighbors), TissueRegionMixup (cross-region prototype mixing) | Perturb features along the tissue manifold using spatial structure |
| **Feature-only perturbation** | Gaussian noise, feature dropout | Channel-level noise |
| **Coordinate transforms** | CoordAffine (rotation, flip, scale), SpatialGridShuffle, coordinate jitter | Change spatial layout without modifying features |

View generators compose these into SSL-appropriate pipelines: `DualViewAugmentor` (two independent views for contrastive/BYOL/VICReg), `MultiCropAugmentor` (2 global + N local views for DINO). JEPA batching handles masking at the collator level.

### Stage 5 — Evaluate

| | |
|---|---|
| **Goal** | Compute comprehensive metrics on held-out predictions |
| **Entry** | `scripts/evaluate.py` |
| **Core modules** | `oceanpath.eval.core`, `oceanpath.eval.plots` |
| **Reads** | OOF predictions parquet |
| **Writes** | `metrics.json`, ROC/PR curves, probability histograms, calibration plots |
| **Techniques** | AUROC, accuracy, balanced accuracy, F1, MCC, Cohen's kappa; bootstrap confidence intervals (2000 resamples); ECE + Brier calibration; operating-point analysis (Youden's J); threshold stability grading; patient-level aggregation |

### Stage 6 — Analyze

| | |
|---|---|
| **Goal** | Post-hoc analysis: attention maps, embedding visualization, model comparison |
| **Entry** | `scripts/analyze.py` |
| **Core modules** | `oceanpath.eval.comparison`, `oceanpath.eval.attention`, `oceanpath.eval.umap_viz` |
| **Reads** | Predictions, embeddings, attention weights |
| **Writes** | Statistical comparison JSON, UMAP plots, attention heatmaps |
| **Techniques** | DeLong test (AUC difference), McNemar test (paired predictions), bootstrap paired difference; UMAP embedding visualization |

### Stage 7 — Export & Serve

| | |
|---|---|
| **Goal** | Package the model for deployment |
| **Entry** | `scripts/export_model.py`, `scripts/serve.py` |
| **Core modules** | `oceanpath.serving` |
| **Reads** | Best checkpoint |
| **Writes** | `model.ts` (TorchScript), `model.onnx` (ONNX) |
| **Techniques** | Trace-based TorchScript export; ONNX export with dynamic axes; FastAPI + ONNX Runtime inference server |

## MIL Aggregators

All aggregators inherit `BaseMIL`, implement `forward_features(h, mask, coords, return_attention)`, and produce a standardized `MILOutput(slide_embedding, logits, extras)`.

| Architecture | Key Idea | Complexity | Dependency |
|---|---|---|---|
| **ABMIL** | Gated attention weighted sum | O(N) | — |
| **TransMIL** | Nystrom attention + PPEG + CLS token | O(N) | `nystrom-attention` |
| **Static** | Mean/max pooling baseline | O(N) | — |
| **Perceiver** | Learned latent tokens cross-attend to patches | O(M*N), M << N | — |
| **Multihead ABMIL** | K independent gated attention heads, concat + project | O(K*N) | — |
| **BiMamba-2** | Bidirectional Mamba-2 SSM + masked mean pool | O(N) | `mamba-ssm` (CUDA) |

## Typical Commands

```bash
# ── Full pipelines ────────────────────────────────────────────────────────

# Supervised: run all 7 stages
python scripts/pipeline.py pipeline_profile=supervised \
  data=gej encoder=univ1 model=abmil

# SSL pretraining: run all 4 stages
python scripts/pipeline.py pipeline_profile=pretrain_only \
  data=uni2h_pretrain encoder=uni2h model=abmil pretrain_training=vicreg

# Dry run (validate DAG, print Mermaid, don't execute)
python scripts/pipeline.py ... pipeline.dry_run=true

# ── Individual stages ─────────────────────────────────────────────────────

# Train with overrides
python scripts/train.py data=gej encoder=univ1 model=perceiver \
  splits=kfold5 training.lr=1e-4 training.max_epochs=80

# Pretrain with BYOL
python scripts/pretrain.py data=uni2h_pretrain encoder=uni2h \
  model=abmil pretrain_training=byol

# Fine-tune from pretrained aggregator
python scripts/train.py data=gej encoder=uni2h model=abmil \
  training.aggregator_weights_path=outputs/pretrain/.../best.ckpt \
  training.freeze_aggregator=false

# Export
python scripts/export_model.py \
  --checkpoint outputs/train/.../fold_0/best.ckpt \
  --output-dir outputs/export --feature-dim 1024

# ── Pretraining data prep ──────────────────────────────────────────────────

# Curate pretraining manifest (filter sources, discard small slides)
python scripts/pretraining_data_cleanining.py

# Build mmap with spatial-stratified capping + coverage QC
python scripts/build_mmap.py data=uni2h_pretrain encoder=uni2h \
  storage.max_instances=16384 storage.cap_strategy=spatial_stratified

# Visualize coverage of capped slides
python scripts/visualize_mmap_coverage.py \
  --h5_dir /path/to/h5 --mmap_dir /path/to/mmap --only_capped

# Benchmark batching strategies
python scripts/benchmark_batching.py --mmap_dir /path/to/mmap

# Compare two experiments
python scripts/compare_experiments.py \
  --pred-a outputs/model_a_preds.csv \
  --pred-b outputs/model_b_preds.csv \
  --output outputs/significance.json
```

## Mental Model

```
                     ┌─────────────────────────────────────────────┐
                     │              Hydra Configs                  │
                     │  (data + encoder + model + training + ...)  │
                     └────────────┬────────────────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   PipelineRunner (DAG)  │
                     │  topo-sort → freshness  │
                     │  check → run or skip    │
                     └────────────┬────────────┘
                                  │
        ┌─────────────────────────┼───────────────────────────┐
        ▼                         ▼                           ▼
   ┌─────────┐            ┌──────────────┐            ┌────────────┐
   │  Data    │            │   Models     │            │    SSL     │
   │ mmap,    │            │ BaseMIL →    │            │ 5 losses,  │
   │ datasets,│───────────▶│ registry →   │◀───────────│ dual-view  │
   │ splits   │            │ WSIClassifier│            │ augment    │
   └─────────┘            └──────────────┘            └────────────┘
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼               ▼
              ┌──────────┐ ┌──────────┐   ┌────────────┐
              │ Training │ │   Eval   │   │   Export   │
              │ Lightning│ │ metrics, │   │ TorchScript│
              │ module   │ │ CI, stats│   │ ONNX, API  │
              └──────────┘ └──────────┘   └────────────┘
```

**Key invariants:**
- Every stage declares its inputs, outputs, and config keys. The DAG engine handles ordering, caching, and invalidation.
- All aggregators conform to the `BaseMIL` contract: `[B, N, D] → [B, E]` with optional mask, coords, and attention output.
- SSL pretraining produces aggregator weights that plug directly into supervised fine-tuning via `training.aggregator_weights_path`.
- Atomic transactions mean a crashed run never leaves partial outputs — either a stage fully succeeds or it rolls back.
