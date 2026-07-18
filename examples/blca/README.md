# BLCA reference integration

BLCA demonstrates the complete foundation pipeline for a small binary grading task.
It is an integration profile, not special-cased library behavior.

1. Set `OCEANPATH_ROOT`.
2. Put slides in `$OCEANPATH_ROOT/slides/blca/`.
3. Copy `manifest.example.csv` to `$OCEANPATH_ROOT/manifests/blca.csv` and replace
   the example rows.
4. Inspect each stage with `--cfg job --resolve`.
5. Run `scripts/pipeline.py` or the individual stage scripts.

The default `kfold5` profile uses `De ID` for patient grouping and
`Binary WHO 2022` for stratification. The optional `blca_custom` profile reserves
WHO 1973 grade 2 for the test set and uses the remaining cases for cross-validation.
