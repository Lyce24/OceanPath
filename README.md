# OceanPath

[![CI](https://github.com/Lyce24/OceanPath/actions/workflows/ci.yml/badge.svg)](https://github.com/Lyce24/OceanPath/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)

Self-supervised pretraining of slide-level multiple-instance learning (MIL) aggregators for
computational pathology, and the pipeline around it: feature stores, supervised fine-tuning,
label-efficiency probes, export, and a CPU serving image.

OceanPath pretrains Mamba-2 MIL aggregators without labels on UNI2-h patch features of
**14,529 public TCGA and CPTAC whole-slide images (142.7 million patches)** with four
slide-level objectives (VICReg, JEPA, LeJEPA, LeJEPA-MC), then measures what that pretraining
buys on public downstream tasks against strong supervised baselines: TCGA-NSCLC subtyping with
CPTAC-NSCLC as an external cohort, colorectal BRAF mutation, and PANDA prostate grading.

<p align="center">
  <img src="docs/assets/ssl_views_tcga_luad.png" width="70%" alt="Spatial coverage of the SSL view strategies on one TCGA lung adenocarcinoma slide" />
</p>

## Contents

- [Results](#results): fine-tuning, label efficiency, pretraining compute, scaling, serving latency
- [Install](#install) · [Data layout](#data-layout) · [Reproduce](#reproduce) · [Serve](#serve)
- [Pretrained weights](#pretrained-weights) · [Documentation](#documentation) · [License](#license)

## Results

Every table below is regenerated from finished runs by `make results` (with `OCEANPATH_ROOT`
set); each row names the command that produced it. Patient-level metrics throughout. Folds are
fixed (5-fold, patient-grouped, stratified); the seed changes initialisation, data order and,
for probes, which patients are labelled.

### Fine-tuning

ABMIL and Mamba-2 MIL trained from scratch, and Mamba-2 MIL fine-tuned from each SSL checkpoint
(aggregator learning rate 1e-5, head 1e-4). Mean ± SD over seeds of the pooled out-of-fold
metric, a 95% patient-bootstrap CI of that seed mean, and the difference of seed means to
Mamba-2 MIL from scratch with a paired bootstrap CI. CPTAC-NSCLC is scored by each seed's five
TCGA fold models, averaged.
The SSL checkpoints were chosen with TCGA-NSCLC and TCGA-BRAF labels
([checkpoint selection](#pretraining-compute-and-checkpoint-selection)), so their rows on those two
tasks lean optimistic; PANDA played no part in choosing them.

<!-- results:supervised -->
*Pending: the three-seed runs are not finished; `make results` fills this table from them.*
<!-- /results:supervised -->

### Label efficiency

Frozen slide embeddings, logistic-regression probes trained on 1–100% of the training patients
(three random draws per fraction; at 100% one), against three pooling baselines and a
randomly initialised Mamba-2 aggregator with the same architecture.

<!-- results:label-efficiency -->
*Pending: the sweeps are not finished; `make results` fills the curves and tables from them.*
<!-- /results:label-efficiency -->

### Pretraining compute and checkpoint selection

From the Weights & Biases run records (`scripts/pretraining_compute.py`; every run in
`results/pretraining/compute.csv`). GPU-hours are wall-clock hours times GPUs.

| Objective | Hardware | Epochs run | Checkpoint epoch | Wall-clock | GPU-hours |
| --- | --- | --- | --- | --- | --- |
| VICReg | 1× RTX A5000 (workstation) | 100 | 10 | 26.0 h | 26.0 |
| JEPA | 2× RTX A5000 (Brown CCV Oscar) | 100 | 65 | 10.1 h | 20.1 |
| LeJEPA | 2× RTX A5500 (Brown CCV Oscar) | 100 | 65 | 9.2 h | 18.3 |
| LeJEPA-MC | 2× RTX 3090 (Brown CCV Oscar) | 100 | 35 | 10.5 h | 21.1 |
| **The four runs** | | | | | **85.5** |
| All 85 logged pretraining runs (development, ablations, failures) | | | | | **602** (Oscar 336, workstation 266) |

Each run trained for 100 epochs, but the checkpoint used downstream is not the last one. Every
five epochs the run fit linear probes on labelled cohorts and kept the checkpoint with the best
weighted score: TCGA-BRAF AUROC (weight 0.35–0.40), TCGA-NSCLC AUROC (0.15–0.25) and an
in-house gastro-oesophageal cohort (0.40–0.45). The epochs above are those checkpoints
(0-indexed, as logged). Labels from two of the evaluation tasks therefore helped pick the SSL
weights. Treat the SSL rows on TCGA-NSCLC and TCGA-BRAF as optimistic. PANDA was never part of
the score, and CPTAC-NSCLC was only reported, never weighted.

### Memory and throughput versus bag size

One slide per step, bf16 autocast, UNI2-h features (1,536 dims), RTX A5000 (24 GB)
(`scripts/benchmark_scaling.py`). Attention pooling, the Perceiver, TransMIL's Nyström attention
and Mamba-2 scale linearly in the number of patches; the TITAN-style ViT builds an N × N
distance matrix for its 2-D ALiBi bias and runs out of memory past 16,384 patches. A Mamba-2 MIL
training step on a 131,072-patch slide peaks at 5.2 GB, under a quarter of the card, at 1.2
million patches per second. Small bags leave the GPU waiting on kernel launches, so throughput
rises with N.

<p align="center">
  <img src="results/scaling/scaling.png" width="90%" alt="Peak GPU memory and throughput versus patches per slide" />
</p>

<!-- results:scaling -->
#### Training step (peak memory · patches per second)

| Aggregator | N = 1,024 | N = 2,048 | N = 4,096 | N = 8,192 | N = 16,384 | N = 32,768 | N = 65,536 | N = 131,072 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean pooling + MLP | 0.04 GB · 268k/s | 0.06 GB · 460k/s | 0.09 GB · 814k/s | 0.15 GB · 1,967k/s | 0.26 GB · 3,252k/s | 0.45 GB · 9,592k/s | 0.84 GB · 12,613k/s | 1.65 GB · 13,445k/s |
| ABMIL | 0.05 GB · 184k/s | 0.07 GB · 245k/s | 0.10 GB · 738k/s | 0.17 GB · 1,458k/s | 0.31 GB · 2,592k/s | 0.56 GB · 5,354k/s | 1.09 GB · 6,711k/s | 2.16 GB · 6,914k/s |
| Multi-head ABMIL | 0.10 GB · 58k/s | 0.13 GB · 192k/s | 0.19 GB · 373k/s | 0.31 GB · 502k/s | 0.55 GB · 739k/s | 1.02 GB · 1,872k/s | 1.97 GB · 2,535k/s | 3.88 GB · 2,636k/s |
| Perceiver | 0.48 GB · 29k/s | 0.49 GB · 62k/s | 0.51 GB · 123k/s | 0.57 GB · 221k/s | 0.75 GB · 447k/s | 1.13 GB · 923k/s | 1.97 GB · 1,444k/s | 3.64 GB · 2,218k/s |
| TransMIL | 0.18 GB · 27k/s | 0.26 GB · 59k/s | 0.39 GB · 120k/s | 0.66 GB · 212k/s | 1.22 GB · 297k/s | 2.35 GB · 317k/s | 4.57 GB · 298k/s | 9.08 GB · 280k/s |
| ViT, 2-D ALiBi | 0.58 GB · 29k/s | 0.76 GB · 56k/s | 1.56 GB · 45k/s | 4.71 GB · 27k/s | 17.57 GB · 15k/s | OOM | — | — |
| Mamba-2 MIL | 0.11 GB · 31k/s | 0.15 GB · 64k/s | 0.23 GB · 96k/s | 0.39 GB · 208k/s | 0.70 GB · 447k/s | 1.34 GB · 928k/s | 2.63 GB · 1,078k/s | 5.20 GB · 1,161k/s |
| Bidirectional Mamba | 0.19 GB · 18k/s | 0.26 GB · 40k/s | 0.39 GB · 82k/s | 0.65 GB · 154k/s | 1.16 GB · 307k/s | 2.22 GB · 581k/s | 4.32 GB · 606k/s | 8.52 GB · 613k/s |

#### Inference (peak memory · patches per second)

| Aggregator | N = 1,024 | N = 2,048 | N = 4,096 | N = 8,192 | N = 16,384 | N = 32,768 | N = 65,536 | N = 131,072 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean pooling + MLP | 0.03 GB · 1,248k/s | 0.04 GB · 2,545k/s | 0.06 GB · 5,053k/s | 0.11 GB · 13,502k/s | 0.19 GB · 18,412k/s | 0.33 GB · 19,325k/s | 0.64 GB · 21,784k/s | 1.26 GB · 22,898k/s |
| ABMIL | 0.02 GB · 870k/s | 0.03 GB · 1,777k/s | 0.05 GB · 3,451k/s | 0.09 GB · 7,289k/s | 0.17 GB · 13,395k/s | 0.33 GB · 15,862k/s | 0.64 GB · 17,517k/s | 1.27 GB · 9,665k/s |
| Multi-head ABMIL | 0.04 GB · 249k/s | 0.04 GB · 512k/s | 0.06 GB · 1,024k/s | 0.10 GB · 3,014k/s | 0.18 GB · 6,011k/s | 0.34 GB · 8,075k/s | 0.65 GB · 8,355k/s | 1.27 GB · 8,762k/s |
| Perceiver | 0.16 GB · 128k/s | 0.17 GB · 255k/s | 0.18 GB · 500k/s | 0.24 GB · 1,092k/s | 0.37 GB · 2,158k/s | 0.64 GB · 4,366k/s | 1.17 GB · 5,688k/s | 2.23 GB · 6,151k/s |
| TransMIL | 0.07 GB · 74k/s | 0.10 GB · 207k/s | 0.16 GB · 409k/s | 0.29 GB · 769k/s | 0.54 GB · 1,102k/s | 1.05 GB · 1,188k/s | 2.04 GB · 1,105k/s | 4.05 GB · 1,008k/s |
| ViT, 2-D ALiBi | 0.25 GB · 197k/s | 0.45 GB · 215k/s | 1.24 GB · 144k/s | 4.44 GB · 84k/s | 17.26 GB · 45k/s | OOM | — | — |
| Mamba-2 MIL | 0.04 GB · 41k/s | 0.06 GB · 124k/s | 0.10 GB · 120k/s | 0.17 GB · 240k/s | 0.31 GB · 541k/s | 0.59 GB · 1,094k/s | 1.17 GB · 1,523k/s | 2.31 GB · 2,483k/s |
| Bidirectional Mamba | 0.07 GB · 138k/s | 0.10 GB · 269k/s | 0.15 GB · 532k/s | 0.24 GB · 905k/s | 0.44 GB · 1,343k/s | 0.84 GB · 1,923k/s | 1.62 GB · 1,934k/s | 3.19 GB · 1,947k/s |
<!-- /results:scaling -->

### Serving latency (CPU)

p50 / p95 milliseconds per slide through the server's inference backend, and the HTTP round
trip of the serving image's endpoints (`scripts/benchmark_serving.py`; Xeon W-2295, pinned
cores; random weights, which do not change latency). ABMIL answers a 2,000-patch slide in
12–17 ms on four threads with any backend. JSON bodies cost about ten bytes per value (1.3 s for
2,000 patches), so large bags go through `POST /predict/npy`; past ~8,000 patches the upload
itself dominates, and sending the bag as float16 halves it. TransMIL's ONNX and TorchScript
exports and the Perceiver's TorchScript export fail the exporter's parity check, so those
models are served from the PyTorch backend only.

<!-- results:serving -->
#### 1 CPU thread (p50 / p95 ms)

| Aggregator | Path | N = 500 | N = 2,000 | N = 8,000 | N = 32,000 |
| --- | --- | --- | --- | --- | --- |
| abmil | pytorch | 11.7 / 13.0 | 38.6 / 44.2 | 176.0 / 185.7 | 767.8 / 1,016.3 |
| abmil | torchscript | 10.1 / 10.7 | 37.9 / 43.1 | 181.9 / 247.2 | 769.0 / 823.1 |
| abmil | onnx | 10.1 / 10.7 | 40.4 / 50.3 | 197.1 / 209.7 | 850.7 / 914.7 |
| abmil | HTTP npy (onnx) | 24.9 / 32.4 | 63.0 / 68.6 | 420.6 / 507.0 | 5,959.6 / 6,556.9 |
| abmil | HTTP json (onnx) | 289.2 / 302.2 | 1,415.8 / 1,529.9 | — | — |
| perceiver | pytorch | 138.4 / 163.9 | 163.7 / 199.6 | 285.4 / 320.7 | 924.1 / 1,039.6 |
| perceiver | onnx | 136.1 / 165.6 | 176.5 / 203.7 | 389.8 / 422.3 | 1,216.7 / 1,392.6 |
| perceiver | HTTP npy (onnx) | 155.6 / 183.9 | 229.4 / 254.4 | 552.8 / 703.1 | 4,066.2 / 4,529.0 |
| perceiver | HTTP json (onnx) | 429.8 / 464.5 | 1,632.4 / 1,942.4 | — | — |
| transmil | pytorch | 143.8 / 152.8 | 288.5 / 328.2 | 1,427.6 / 1,504.3 | 6,765.0 / 6,785.5 |

#### 4 CPU threads (p50 / p95 ms)

| Aggregator | Path | N = 500 | N = 2,000 | N = 8,000 | N = 32,000 |
| --- | --- | --- | --- | --- | --- |
| abmil | pytorch | 4.4 / 7.1 | 15.4 / 24.5 | 75.7 / 93.9 | 296.0 / 340.1 |
| abmil | torchscript | 4.1 / 4.4 | 16.5 / 25.0 | 65.2 / 89.8 | 291.5 / 317.0 |
| abmil | onnx | 3.3 / 3.6 | 12.5 / 27.8 | 69.8 / 111.7 | 283.7 / 350.4 |
| abmil | HTTP npy (onnx) | 16.9 / 18.7 | 43.7 / 55.1 | 196.9 / 263.2 | 2,042.6 / 3,358.6 |
| abmil | HTTP json (onnx) | 319.3 / 351.0 | 1,331.1 / 1,754.4 | — | — |
| perceiver | pytorch | 61.0 / 73.3 | 75.5 / 80.5 | 137.4 / 154.1 | 426.0 / 454.1 |
| perceiver | onnx | 44.1 / 46.3 | 57.0 / 68.0 | 139.5 / 198.2 | 428.7 / 576.7 |
| perceiver | HTTP npy (onnx) | 59.6 / 74.5 | 103.7 / 119.7 | 417.1 / 533.0 | 3,835.4 / 5,035.3 |
| perceiver | HTTP json (onnx) | 362.0 / 379.4 | 1,573.0 / 1,667.5 | — | — |
| transmil | pytorch | 45.0 / 64.4 | 106.1 / 122.1 | 567.5 / 601.2 | 2,494.8 / 2,621.1 |
<!-- /results:serving -->

## Install

OceanPath uses [uv](https://docs.astral.sh/uv/) and the committed `uv.lock` (Python 3.10–3.12).

| Use | Command | Needs |
| --- | --- | --- |
| Everything except the Mamba-2 kernels (training of attention models, probes, export) | `make sync` | CPU is enough |
| Mamba-2 aggregators | `make sync-gpu` | NVIDIA GPU and `nvcc`; compiles `mamba-ssm` and `causal-conv1d` once (about 25 min) |
| Feature extraction from whole-slide images | `uv sync --extra extract` | TRIDENT, OpenSlide |
| Serving only, without torch | `docker build -f docker/serve.Dockerfile -t oceanpath-serve .` | Docker |

`mamba-ssm` publishes no wheels for torch 2.8 or later, so the `mamba` extra builds its kernels
against the locked torch. All Mamba imports are guarded: without the extra those models are
simply not registered, and everything else installs and tests on a laptop. `make ci` runs the
same lint, lockfile, test and build steps as GitHub Actions.

## Data layout

Every storage root derives from one environment variable (`configs/platform/local.yaml`);
`configs/platform/example.yaml` is a template for machines whose slides, features and outputs
live in different places.

```text
$OCEANPATH_ROOT/
  features/UNI2_h_features/{TCGA,CPTAC,PANDA}/   per-slide patch features (H5)
  manifests/                                     label and pretraining manifests
  splits/<dataset>/kfold5_seed42/                fold assignments
  mmap/<dataset>/                                chunked feature stores
  checkpoints/aggregator_weights_<method>.pt     SSL checkpoints for fine-tuning and probes
  outputs/                                       runs, sweeps, reports
```

The public-cohort manifests were derived from TCGA/CPTAC metadata and the PANDA challenge
labels; UNI2-h features come from MahmoodLab's gated release.

## Reproduce

```bash
export OCEANPATH_ROOT=/path/to/oceanpath-data

# Feature stores and folds (once per dataset)
python scripts/build_mmap.py data=tcga_nsclc encoder=uni2h storage.filter_by_csv=true
python scripts/make_splits.py data=tcga_nsclc encoder=uni2h splits=kfold5

# Pretraining (Slurm template; one objective per job)
sbatch slurm/pretrain.sbatch model=mamba2mil pretrain_training=lejepa_2c

# Fine-tuning tables: tasks x initialisations x seeds
bash scripts/run_supervised_baselines.sh --task nsclc --seeds "42 43 44"

# Label efficiency: pooling baselines, random init and every SSL checkpoint
bash scripts/run_label_efficiency.sh --task nsclc_xfer \
  --ssl-ckpt vicreg=$OCEANPATH_ROOT/checkpoints/aggregator_weights_vicreg.pt \
  --ssl-ckpt lejepa=$OCEANPATH_ROOT/checkpoints/aggregator_weights_lejepa.pt

# Systems and serving benchmarks
python scripts/benchmark_scaling.py --out results/scaling
python scripts/benchmark_serving.py --out results/serving_latency

# Tables, figures and this README's result blocks
make results
```

`slurm/` has job templates for pretraining, fine-tuning and the probe sweep. On this
project's workstation only one CUDA process may run at a time (concurrent processes trigger
driver faults under WSL2), so the drivers take `--jobs 1` there.

## Serve

`scripts/export_model.py` exports a trained checkpoint to ONNX and TorchScript, checks both
against PyTorch on several bag sizes, and writes a model card; an export whose check fails or
cannot run is reported as failed. The CPU image serves the ONNX artifact with no torch
installed:

```bash
make demo-artifact                       # random-weight ABMIL, for trying the API
docker build -f docker/serve.Dockerfile -t oceanpath-serve .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts/demo:/models:ro" oceanpath-serve

curl -s http://127.0.0.1:8000/health
python - <<'PY'
import io, numpy as np, urllib.request
x = np.random.randn(8000, 1536).astype(np.float32)        # one slide's patch features
buf = io.BytesIO(); np.save(buf, x)
print(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/predict/npy", buf.getvalue())).read())
PY
```

Endpoints: `GET /health`, `GET /model-card`, `POST /predict` (JSON `[N, D]`),
`POST /predict/npy` (`.npy` body; pickled arrays are refused) and `POST /predict/batch`.
Malformed bags (wrong feature dimension, empty, non-finite) are 400s. Without Docker:
`python -m oceanpath.serving --artifact-dir artifacts/demo --threads 4`.

## Pretrained weights

The four Mamba-2 MIL checkpoints are not published yet. UNI2-h and its feature release are
licensed CC-BY-NC-ND 4.0, and MahmoodLab counts models trained on their outputs as derivatives
that may not be used commercially without approval. `scripts/publish_pretrained_weights.py`
stages a gated, non-commercial Hugging Face release (safetensors, config, model card) for when
that is cleared.

## Documentation

- [Pipeline reference](docs/pipeline.md): stages, the DAG engine, SSL objectives and
  augmentations, aggregators, linear probing and its robustness rules, typical commands
- [Slurm templates](slurm/README.md)
- `configs/`: Hydra groups for data, encoders, models, training, SSL objectives, splits and platforms

## License

Code: MIT. Model weights trained on UNI2-h features follow UNI2-h's CC-BY-NC-ND 4.0 terms.
