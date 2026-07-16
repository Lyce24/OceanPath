# Pipeline DAGs With Fingerprints

The pipeline module provides a DAG builder and Mermaid rendering with per-stage fingerprints:

- `build_supervised_pipeline(cfg)`

Use:

```python
from oceanpath.pipeline import build_supervised_pipeline

runner = build_supervised_pipeline(cfg)
print(runner.render_dag(target="export_and_serve", cfg=cfg, include_fingerprint=True))
```

Each node label includes `fp:<12-hex>` and each stage execution writes transaction metadata for freshness checks.

CLI runner:

```bash
# Supervised 7-stage DAG
python scripts/pipeline.py pipeline_profile=supervised \
  platform=local data=gej encoder=univ1 model=abmil
```
