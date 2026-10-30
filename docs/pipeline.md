# Pipeline reference

How OceanPath's stages fit together, what each reads and writes, and the commands that
drive them. For the results and how to reproduce them, start from the [README](../README.md).

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
| **Core modules** | `oceanpath.ssl.modules.pretrain_module`, `oceanpath.ssl.modules.losses`, `oceanpath.ssl.data.augmentation`, `oceanpath.ssl.data.dataset`, `oceanpath.ssl.data.datamodule` |
| **Reads** | MMap store, split manifest |
| **Writes** | Aggregator checkpoint, training logs |
| **Techniques** | Four slide-level SSL methods, dual-view + multi-crop augmentation, quality monitoring via RankMe (effective rank), alpha-ReQ (power-law exponent), and z-dim std |

**SSL methods**: **VICReg** (variance-invariance-covariance), **JEPA** (symmetric joint-embedding predictive with EMA target), **LeJEPA-2C** (centroid + SIGReg, two views), **LeJEPA-MC** (multi-crop centroid + SIGReg, V_g globals + V_l locals).

#### Batching

Every slide in a pretraining batch is cut to the same number of patches by its augmentor,
so collation is a plain stack and the SSL module never sees padding
(`oceanpath.ssl.data.datamodule`):

| Mode | Methods | Collated batch |
| --- | --- | --- |
| Dual-view (`stack_collate`) | VICReg, JEPA, LeJEPA | `view1`, `view2`: `[B, fixed_n, D]` each |
| Multi-crop (`multicrop_stack_collate`) | LeJEPA-MC | `global_views`: `[V_g, B, N_g, D]`, `local_views`: `[V_l, B, N_l, D]` |

`fixed_n`, the view counts and their sizes are set per method in
`configs/pretrain_training/{method}.yaml`. Slides are pre-capped when they are read
(`PretrainDataset`: random, contiguous, head or spatially stratified), and a pre-cap
smaller than `fixed_n` is refused.

#### Augmentation Pipeline

Feature-space augmentations are organized into four categories, ordered from strongest to subtlest:

| Category | Transforms | Purpose |
|---|---|---|
| **Patch selection** (which patches each view sees) | Spatial crop, density-aware subsampling, region drop, instance dropout, random subsampling | Primary source of view diversity — different spatial coverage per view |
| **Coords-aware feature perturbation** | LocalFeatureSmooth (k-NN blending), SpatialFeatureInterpolation (lerp toward neighbors), TissueRegionMixup (cross-region prototype mixing) | Perturb features along the tissue manifold using spatial structure |
| **Feature-only perturbation** | Gaussian noise, feature dropout | Channel-level noise |
| **Coordinate transforms** | CoordAffine (rotation, flip, scale), SpatialGridShuffle, coordinate jitter | Change spatial layout without modifying features |

View generators compose these into SSL-appropriate pipelines: `WSIDualViewAugmentor` (two views for VICReg, JEPA and LeJEPA) and `WSIMultiCropAugmentor` (V_g global + V_l local views for LeJEPA-MC).

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
| **Mamba-2 MIL** (`mamba2mil`) | Mamba-2 SSD blocks + gated attention pooling (optional COBRA-style pooling of raw embeddings at inference, off by default) | O(N) | `mamba-ssm` (CUDA) |
| **ViT** (`vit`, TITAN-style) | Transformer over patch tokens with 2-D ALiBi from coordinates | O(N²) | — |

## Linear Probing & Label Efficiency

Frozen-encoder evaluation lives in `scripts/linear_probing.py` (Hydra) and the sklearn pipeline in `oceanpath.modules.linear_probing_sklearn` (StandardScaler → LogisticRegression with patient-grouped CV or external-test protocol). Embeddings are extracted once into a slide-level mmap (`embedding.mode ∈ {mean_pool, max_pool, mean_max_pool, aggregator}`) and reused across runs via a fingerprint-keyed cache.

### Label efficiency sweep

`scripts/run_label_efficiency.sh` runs the full Monte-Carlo sweep over labeled-data fractions to test whether SSL gains hold under low-label settings. Defaults:

| Axis | Default | Override |
|---|---|---|
| Fractions | `0.01 0.05 0.10 0.25 0.50 1.00` | `--fractions "0.01 0.10 1.00"` |
| Seeds | `42 43 44` (Monte Carlo over which patients are labeled) | `--seeds "42 43 44 45 46"` |
| Tasks | `nsclc_cv  nsclc_xfer  braf  panda  gej` | `--task <tag>` |
| Modes | `mean_pool  max_pool  mean_max_pool  aggregator_random  ssl` | `--mode <tag>` |
| Probe tuning | `inner_splits=3  c_grid=[0.01,0.1,1.0,10.0]` | `--inner-splits 0 --c-grid '[1.0]'` |
| SSL checkpoint | none (skips `mode=ssl` unless provided) | `--ssl-ckpt path/to/best.ckpt` |

```bash
# Full sweep (5 tasks × 5 modes × 6 fractions × 3 seeds = 450 runs)
bash scripts/run_label_efficiency.sh --ssl-ckpt outputs/pretrain/.../best.ckpt

# Quick check: one fraction, one mode, one seed, dry-run
bash scripts/run_label_efficiency.sh --dry-run \
  --task nsclc_cv --mode mean_pool --fraction 0.10 --seeds 42

# Target the 1% regime with extra seeds for tighter error bars
bash scripts/run_label_efficiency.sh --fractions "0.01 0.05" --seeds "42 43 44 45 46"
```

Outputs land at `$SWEEP_ROOT/<TS>/frac_<ftag>/<task>__<mode>/seed_<N>/<cache_tag>/`, plus a `sweep_summary.csv` recording status/duration per `(task, mode, fraction, seed)`.

### Robustness contract

The probe pipeline is built so the same protocol applies cleanly across all fractions:

- **Patient-grouped subsampling** ([`subsample_labeled_training_data`](../src/oceanpath/modules/linear_probing_sklearn.py)) — all slides for a patient stay together; per-class quotas with `ceil(N·f)` ensure every class is represented even at 1%; missing classes after rounding are backfilled.
- **Validation/test never subsampled** — only the inner training split is touched. External test cohorts and held-out CV folds remain full-size.
- **Per-fold seed variation** — within a single run, fold `k` uses `base_seed + k` so each fold sees a different subsample. Across the sweep, the outer `--seeds` loop adds Monte Carlo variance over subsamples.
- **Inner-CV graceful fallback** — `select_best_c` automatically skips inner CV and uses the C-grid midpoint when there aren't enough groups/classes (the typical case at 1%). The same `inner_splits=3 c_grid=[0.01,0.1,1.0,10.0]` setting is therefore safe across fractions and yields a uniform protocol.
- **Continuous-label fallback** — if labels aren't integer-valued (regression), subsampling falls back to uniform random patient sampling instead of failing on `astype(int)`.
- **Embedding cache reused across fractions and seeds** — extraction depends only on `(mode, arch, ckpt, max_instances, sampling)`. Re-running a sweep does not re-extract.

### Aggregating results

`scripts/collect_lp_results.py` walks summary.json files and aggregates seed_<N> subdirs into mean ± std over seeds (the inter-seed std is the canonical Monte-Carlo error bar for low-label regimes):

```bash
# Latest sweep, seed-aggregated table
python scripts/collect_lp_results.py --root $SWEEP_ROOT --latest

# Per-seed breakdown for inspecting variance
python scripts/collect_lp_results.py --root $SWEEP_ROOT/<TS> --by-seed

# CSV export for plotting
python scripts/collect_lp_results.py --root $SWEEP_ROOT/<TS> --csv label_eff.csv

# Tables, curves and paired differences to the random-init aggregator, as in the README
python scripts/collect_label_efficiency.py --root $OCEANPATH_ROOT/outputs/label_efficiency \
  --out results/label_efficiency
```

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

# Pretrain a Mamba-2 aggregator with LeJEPA
python scripts/pretrain.py data=uni2h_pretrain_2048 encoder=uni2h \
  model=mamba2mil pretrain_training=lejepa_2c

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

# Aggregator memory and throughput versus bag size (GPU)
python scripts/benchmark_scaling.py --out results/scaling

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
   │ mmap,    │            │ BaseMIL →    │            │ 4 SSL      │
   │ datasets,│───────────▶│ registry →   │◀───────────│ objectives │
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
