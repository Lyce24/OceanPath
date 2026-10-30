# Public-cohort label manifests

The label tables behind the README's results. Copy them to `$OCEANPATH_ROOT/manifests/SSL/`,
where the data configs look for them (`configs/data/*.yaml`). Slide paths are relative to the
UNI2-h feature root, without the `.h5` suffix.

| File | Task | Rows | Label |
| --- | --- | --- | --- |
| `tcga_nsclc_labels.csv` | TCGA-NSCLC subtyping | 946 slides, 946 patients | `label`: 0 = LUAD (468), 1 = LUSC (478), from the TCGA project |
| `cptac_nsclc_labels.csv` | CPTAC-NSCLC external test | 453 slides | `label`: 0 = LUAD (241), 1 = LSCC (212), from the CPTAC cohort |
| `TCGA_BRAF_labels_filtered.csv` | TCGA colorectal BRAF mutation | 566 slides, 559 patients | `BRAF_label`: 1 = mutated (60), 0 = wild type (506) |

CPTAC patient IDs for the lung adenocarcinoma slides are the first block of the slide UUID,
so each of those slides is its own patient group.

**PANDA** labels are not redistributed here. Build `train_panda_labels.csv` from the
challenge's `train.csv` (Kaggle, "Prostate cANcer graDe Assessment") by adding
`file_path = PANDA/panda/<image_id>` to its `image_id, data_provider, isup_grade,
gleason_score` columns (10,615 slides).
