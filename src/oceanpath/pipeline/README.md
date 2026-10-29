# Pipeline DAGs With Fingerprints

The pipeline module now supports two DAG builders and Mermaid rendering with per-stage fingerprints:

- `build_supervised_pipeline(cfg)`
- `build_pretraining_pipeline(cfg)`

Use:

```python
from oceanpath.pipeline import build_supervised_pipeline, build_pretraining_pipeline

runner = build_supervised_pipeline(cfg)
print(runner.render_dag(target="export_and_serve", cfg=cfg, include_fingerprint=True))

pre_runner = build_pretraining_pipeline(cfg)
print(pre_runner.render_dag(target="export_and_serve", cfg=cfg, include_fingerprint=True))
```

Each node label includes `fp:<12-hex>` and each stage execution writes transaction metadata for freshness checks.

CLI runner:

```bash
# Supervised 7-stage DAG
python scripts/pipeline.py pipeline_profile=supervised \
  platform=local data=gej encoder=univ1 model=abmil

# SSL pretraining 4-stage DAG
python scripts/pipeline.py pipeline_profile=pretrain_only \
  platform=local data=uni2h_pretrain encoder=uni2h model=abmil \
  pretrain_training=vicreg
```
