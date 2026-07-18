"""Architecture-level tests for the dataset-neutral main-branch foundation."""

import json
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ENCODER_PROFILES = {
    "univ1": ("uni_v1", 1024, "vit_large_patch16_224", "MahmoodLab/UNI", 256, "trident"),
    "conch": ("conch_v1", 512, "conch_ViT-B-16", "MahmoodLab/CONCH", 512, "trident"),
    "conch_v15": (
        "conch_v15",
        768,
        "vit_large_patch16_224",
        "MahmoodLab/conchv1_5",
        512,
        "trident",
    ),
    "resnet50": ("resnet50", 1024, "resnet50", "timm/resnet50.tv_in1k", 256, "trident"),
    "virchow": ("virchow", 2560, "vit_huge_patch14_224", "paige-ai/Virchow", 224, "trident"),
    "virchow2": ("virchow2", 2560, "vit_huge_patch14_224", "paige-ai/Virchow2", 224, "trident"),
    "gigapath": (
        "gigapath",
        1536,
        "vit_giant_patch14_dinov2",
        "prov-gigapath/prov-gigapath",
        256,
        "trident",
    ),
    "hoptimus1": (
        "hoptimus1",
        1536,
        "vit_giant_patch14_reg4_dinov2",
        "bioptimus/H-optimus-1",
        224,
        "trident",
    ),
    "gemma4_e4b": (
        "gemma4-e4b",
        768,
        "gemma4_vision",
        "google/gemma-4-E4B",
        224,
        "pending_gemma4",
    ),
    "gemma4_26b": (
        "gemma4-26b",
        1152,
        "gemma4_vision",
        "google/gemma-4-26B-A4B",
        224,
        "pending_gemma4",
    ),
}


@pytest.mark.parametrize(
    "config_name",
    [
        "extract",
        "build_mmap",
        "make_splits",
        "train",
        "evaluate",
        "export",
        "pipeline",
        "benchmark",
    ],
)
def test_essential_configs_compose_with_blca(config_name):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name)

    OmegaConf.to_container(cfg, resolve=True)
    assert cfg.data.name == "blca"


def test_optional_serve_config_is_focused():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="serve")

    assert cfg.export.artifact_dir is None
    assert cfg.serve.backend == "auto"
    assert "data" not in cfg
    assert "training" not in cfg


def test_local_platform_uses_portable_environment_root(monkeypatch):
    root = "/tmp/oceanpath-reference"
    monkeypatch.setenv("OCEANPATH_ROOT", root)
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="extract")

    assert cfg.platform.project_root == root
    assert cfg.data.slide_dir == f"{root}/slides/blca"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("case.SVS", "case"),
        (r"nested\case.NDPI", "nested/case"),
        ("TCGA.case.8923A151", "TCGA.case.8923A151"),
        ("features/case.h5", "features/case"),
        ("/mounted/slides/case.svs", "case"),
    ],
)
def test_slide_id_normalization_is_one_public_contract(value, expected):
    from oceanpath.contracts import normalize_slide_id

    assert normalize_slide_id(value) == expected


@pytest.mark.parametrize(("profile", "expected"), ENCODER_PROFILES.items())
def test_encoder_profiles_declare_adapter_and_source(profile, expected):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="extract", overrides=[f"encoder={profile}"])

    name, feature_dim, family, source, patch_size, adapter = expected
    assert cfg.encoder.name == name
    assert cfg.encoder.feature_dim == feature_dim
    assert cfg.encoder.family == family
    assert cfg.encoder.source == source
    assert cfg.encoder.adapter == adapter
    assert cfg.encoder.mag == 20
    assert cfg.encoder.patch_size == patch_size
    assert cfg.extraction.mag == cfg.encoder.mag
    assert cfg.extraction.patch_size == cfg.encoder.patch_size


@pytest.mark.parametrize("profile", ["gemma4_e4b", "gemma4_26b"])
def test_experimental_gemma_profiles_fail_before_extraction(tmp_path, profile):
    from oceanpath.extraction import ValidationError, validate_inputs
    from oceanpath.workflows.extraction import build_extraction_config

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="extract",
            overrides=[
                f"encoder={profile}",
                f"data.slide_dir={tmp_path}",
                f"data.feature_job_dir={tmp_path / 'features'}",
            ],
        )

    with pytest.raises(ValidationError, match="not implemented"):
        validate_inputs(build_extraction_config(cfg), ["feat"], create_job_dir=False)


def test_remote_encoder_source_changes_fingerprint(tmp_path):
    from oceanpath.extraction import TridentExtractionConfig, compute_encoder_fingerprint

    base = {
        "wsi_dir": str(tmp_path / "slides"),
        "job_dir": str(tmp_path / "features"),
        "patch_encoder": "uni_v1",
    }
    first = compute_encoder_fingerprint(
        TridentExtractionConfig(**base, encoder_source="MahmoodLab/UNI")
    )
    second = compute_encoder_fingerprint(
        TridentExtractionConfig(**base, encoder_source="other/UNI")
    )

    assert first != second


def test_extraction_dry_run_does_not_create_output(tmp_path):
    from oceanpath.extraction import TridentExtractionConfig, run_pipeline

    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    (slide_dir / "reference.svs").touch()
    job_dir = tmp_path / "features" / "blca"
    spec = TridentExtractionConfig(
        wsi_dir=str(slide_dir),
        job_dir=str(job_dir),
        wsi_ext=[".svs"],
    )

    assert run_pipeline(spec, tasks=["seg"], dry_run=True) is None
    assert not job_dir.exists()


def test_extraction_fails_when_expected_features_are_missing(tmp_path, monkeypatch):
    from oceanpath.extraction import TridentExtractionConfig
    from oceanpath.extraction import trident as extraction

    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    (slide_dir / "reference.svs").touch()
    spec = TridentExtractionConfig(
        wsi_dir=str(slide_dir),
        job_dir=str(tmp_path / "features"),
        wsi_ext=[".svs"],
        require_complete=True,
    )
    monkeypatch.setattr(extraction, "create_processor", lambda _cfg: object())
    monkeypatch.setattr(extraction, "run_feature_extraction", lambda _processor, _cfg: None)

    with pytest.raises(RuntimeError, match="Feature extraction is incomplete"):
        extraction.run_pipeline(spec, tasks=["feat"])


def test_external_tracking_is_opt_in():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train")

    assert cfg.wandb.enabled is False


@pytest.mark.parametrize(
    "config_name",
    ["train", "evaluate", "export", "pipeline", "benchmark"],
)
def test_experiment_identity_has_one_profile(config_name):
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name)

    assert cfg.exp_name == cfg.experiment.name
    if "train_dir" in cfg:
        assert cfg.train_dir is None
        paths = FoundationPaths.from_config(cfg)
        assert paths.train_dir.parent == Path(str(cfg.experiment.train_root))
        assert paths.train_dir.name.startswith(f"{cfg.exp_name}_")
    if "eval_dir" in cfg:
        assert cfg.eval_dir is None
        assert FoundationPaths.from_config(cfg).eval_dir.name == cfg.experiment.eval_subdir


def test_blca_task_metadata_drives_evaluation_and_export():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        evaluate_cfg = compose(config_name="evaluate")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        export_cfg = compose(config_name="export")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        benchmark_cfg = compose(config_name="benchmark")

    assert list(evaluate_cfg.eval.class_names) == list(evaluate_cfg.data.class_names)
    assert export_cfg.num_classes == export_cfg.data.num_classes == 2
    assert benchmark_cfg.num_classes == benchmark_cfg.data.num_classes == 2


def test_pipeline_shares_standalone_evaluation_and_export_profiles():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        evaluate_cfg = compose(config_name="evaluate")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        export_cfg = compose(config_name="export")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        pipeline_cfg = compose(config_name="pipeline")

    assert OmegaConf.to_container(pipeline_cfg.eval, resolve=True) == OmegaConf.to_container(
        evaluate_cfg.eval,
        resolve=True,
    )
    assert OmegaConf.to_container(pipeline_cfg.export, resolve=True) == OmegaConf.to_container(
        export_cfg.export,
        resolve=True,
    )


def test_shared_evaluation_and_export_settings_can_be_overridden_in_pipeline():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "eval.n_bootstrap=37",
                "export.opset_version=18",
                "export.formats=[torchscript]",
            ],
        )

    assert cfg.eval.n_bootstrap == 37
    assert cfg.export.opset_version == 18
    assert list(cfg.export.formats) == ["torchscript"]


def test_foundation_model_profiles_are_intentionally_small():
    model_dir = Path(__file__).resolve().parents[1] / "configs" / "model"
    assert {path.stem for path in model_dir.glob("*.yaml")} == {
        "abmil",
        "mhabmil",
        "static",
        "transmil",
    }


def test_foundation_paths_are_the_single_layout_authority():
    from oceanpath.config import FoundationPaths

    cfg = OmegaConf.create(
        {
            "exp_name": "blca_reference",
            "platform": {
                "project_root": "/data",
                "output_root": "/data/outputs",
                "splits_root": "/data/splits",
            },
            "data": {
                "name": "blca",
                "slide_dir": "/data/slides/blca",
                "csv_path": "/data/manifests/blca.csv",
                "feature_job_dir": "/data/features/blca",
                "feature_h5_dir": "/data/features/blca/coords/features_uni",
                "mmap_dir": "/data/mmap/blca/uni",
            },
            "encoder": {"name": "uni"},
            "splits": {"name": "kfold5"},
            "train_dir": "/data/outputs/train/blca_reference",
            "eval_dir": "/data/outputs/train/blca_reference/eval",
            "export": {"artifact_dir": "/data/artifacts/blca_reference"},
        }
    )

    paths = FoundationPaths.from_config(cfg)

    assert paths.dataset_name == "blca"
    assert paths.splits_dir == Path("/data/splits/blca/kfold5")
    assert paths.train_dir == Path("/data/outputs/train/blca_reference")
    assert paths.eval_dir == paths.train_dir / "eval"
    assert paths.artifact_dir == Path("/data/artifacts/blca_reference")


def test_training_layout_separates_material_configs_but_not_operational_flags():
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        baseline = compose(config_name="train")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        changed_model = compose(config_name="train", overrides=["model.attn_dim=64"])
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        forced = compose(
            config_name="train",
            overrides=["training.force_rerun=true", "verbose=true"],
        )

    baseline_path = FoundationPaths.from_config(baseline).train_dir
    assert FoundationPaths.from_config(changed_model).train_dir != baseline_path
    assert FoundationPaths.from_config(forced).train_dir == baseline_path


def test_training_layout_is_identical_across_workflow_roots():
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    resolved_paths = []
    for config_name in ("train", "evaluate", "export", "pipeline"):
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name=config_name)
        resolved_paths.append(FoundationPaths.from_config(cfg).train_dir)

    assert len(set(resolved_paths)) == 1


def test_artifact_identity_ignores_operational_pipeline_controls():
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        baseline = compose(config_name="pipeline")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        operational_override = compose(
            config_name="pipeline",
            overrides=[
                "pipeline.dry_run=true",
                "pipeline.force=true",
                "pipeline.show_dag=false",
                "verbose=true",
                "runtime.strict_environment=true",
            ],
        )

    assert (
        FoundationPaths.from_config(baseline).artifact_dir
        == FoundationPaths.from_config(operational_override).artifact_dir
    )


def test_artifact_identity_changes_with_semantic_export_config():
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        baseline = compose(config_name="pipeline")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        changed = compose(
            config_name="pipeline",
            overrides=["export.formats=[torchscript]"],
        )

    assert (
        FoundationPaths.from_config(baseline).artifact_dir
        != FoundationPaths.from_config(changed).artifact_dir
    )


def test_artifact_identity_changes_with_heldout_evaluation_policy():
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        baseline = compose(config_name="pipeline")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        changed = compose(
            config_name="pipeline",
            overrides=["eval.n_bootstrap=1000"],
        )

    assert (
        FoundationPaths.from_config(baseline).artifact_dir
        != FoundationPaths.from_config(changed).artifact_dir
    )


def test_standalone_and_pipeline_export_share_artifact_path():
    from oceanpath.config import FoundationPaths

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        standalone = compose(config_name="export")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        pipeline = compose(config_name="pipeline")

    assert (
        FoundationPaths.from_config(standalone).artifact_dir
        == FoundationPaths.from_config(pipeline).artifact_dir
    )


def test_workflow_builders_return_typed_domain_inputs():
    from oceanpath.extraction import TridentExtractionConfig
    from oceanpath.splitting import SplitConfig
    from oceanpath.storage import MmapBuildConfig
    from oceanpath.workflows.extraction import build_extraction_config
    from oceanpath.workflows.mmap import build_mmap_config
    from oceanpath.workflows.splitting import build_split_config

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        extraction_cfg = compose(config_name="extract")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        mmap_cfg = compose(config_name="build_mmap")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        split_cfg = compose(config_name="make_splits")

    assert isinstance(build_extraction_config(extraction_cfg), TridentExtractionConfig)
    assert isinstance(build_mmap_config(mmap_cfg), MmapBuildConfig)
    assert isinstance(build_split_config(split_cfg), SplitConfig)


def test_run_context_writes_completion_metadata(tmp_path):
    from oceanpath.runtime import run_context

    cfg = OmegaConf.create({"data": {"csv_path": str(tmp_path / "missing.csv")}})
    with run_context(cfg, stage="test", output_dir=tmp_path):
        pass

    metadata = tmp_path / "_run"
    assert (metadata / "config.yaml").is_file()
    assert (metadata / "provenance.json").is_file()
    status = json.loads((metadata / "status.json").read_text())
    assert status["status"] == "completed"


def test_run_context_records_failure(tmp_path):
    from oceanpath.runtime import run_context

    cfg = OmegaConf.create({})
    with (
        pytest.raises(RuntimeError, match="expected"),
        run_context(cfg, stage="test", output_dir=tmp_path),
    ):
        raise RuntimeError("expected")

    status = json.loads((tmp_path / "_run" / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["exception_type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("scheme", "key", "value", "expected"),
    [
        ("holdout", "n_folds", 5, 1),
        ("custom_kfold", "n_folds", 5, 5),
        ("monte_carlo", "n_repeats", 10, 10),
    ],
)
def test_fold_planner_has_one_explicit_contract(scheme, key, value, expected):
    from oceanpath.training import FoldPlanConfig, plan_folds

    kwargs = {key: value}
    assert len(plan_folds(FoldPlanConfig(scheme=scheme, **kwargs))) == expected


def test_fold_planner_rejects_unknown_scheme():
    from oceanpath.training import FoldPlanConfig, plan_folds

    with pytest.raises(ValueError, match="Unknown split scheme"):
        plan_folds(FoldPlanConfig(scheme="project_specific"))


def test_foundation_contract_does_not_expose_nested_cv():
    from dataclasses import fields
    from inspect import signature

    from oceanpath.datasets import MILDataModule
    from oceanpath.splitting import SUPPORTED_SPLIT_SCHEMES, SplitConfig, get_slide_ids_for_fold
    from oceanpath.training import FoldPlanConfig, plan_folds
    from oceanpath.workflows.training import run_fold

    assert "nested_cv" not in SUPPORTED_SPLIT_SCHEMES
    assert "n_inner_folds" not in {field.name for field in fields(SplitConfig)}
    assert "outer_fold" not in signature(get_slide_ids_for_fold).parameters
    assert "outer_fold" not in signature(MILDataModule).parameters
    assert "outer_fold" not in signature(run_fold).parameters

    with pytest.raises(ValueError, match="Unknown split scheme"):
        SplitConfig(
            scheme="nested_cv",
            name="unsupported",
            csv_path="manifest.csv",
            output_dir="splits",
        )

    with pytest.raises(ValueError, match="Unknown split scheme"):
        plan_folds(FoldPlanConfig(scheme="nested_cv", n_folds=5))

    with pytest.raises(ValueError, match="Unknown split scheme"):
        get_slide_ids_for_fold(
            pd.DataFrame({"slide_id": ["example"]}),
            fold=0,
            scheme="nested_cv",
        )


def test_pipeline_scripts_are_thin_workflow_launchers():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    for script in scripts_dir.glob("*.py"):
        source = script.read_text()
        assert len(source.splitlines()) <= 30, f"{script.name} contains workflow logic"
        assert "oceanpath.workflows" in source, f"{script.name} does not delegate to a workflow"


def test_domain_packages_operate_on_typed_values():
    """Domain packages must not import the config libraries.

    `DictConfig` stops at the workflow boundary: domain code receives typed
    dataclasses and primitive values, never Hydra/OmegaConf objects or helpers.
    """
    package_root = Path(__file__).resolve().parents[1] / "src" / "oceanpath"
    domain_packages = [
        "datasets",
        "eval",
        "export",
        "extraction",
        "models",
        "splitting",
        "storage",
        "training",
        "viz",
    ]
    forbidden_imports = (
        "import hydra",
        "from hydra",
        "import omegaconf",
        "from omegaconf",
        "from oceanpath.config",
        "import oceanpath.config",
    )
    for package_name in domain_packages:
        for module in (package_root / package_name).glob("*.py"):
            source = module.read_text()
            for statement in forbidden_imports:
                assert statement not in source, (
                    f"Config library leaked into domain package: '{statement}' in {module}"
                )


def test_vague_modules_package_stays_dissolved():
    """The old catch-all ``modules`` package must not reappear.

    Lightning mechanics live in ``training`` (typed) and finalization
    orchestration lives in ``workflows`` (config-aware). A new top-level
    ``modules`` package would reintroduce the naming/boundary blind spot.
    """
    package_root = Path(__file__).resolve().parents[1] / "src" / "oceanpath"
    assert not (package_root / "modules").exists(), (
        "oceanpath.modules is back; put Lightning code in training/ and "
        "orchestration in workflows/."
    )


def test_legacy_data_namespace_stays_dissolved():
    package_root = Path(__file__).resolve().parents[1] / "src" / "oceanpath"
    legacy_modules = sorted((package_root / "data").glob("*.py"))
    assert not legacy_modules, (
        "Use the explicit datasets, extraction, storage, and splitting packages; "
        f"do not restore the ambiguous oceanpath.data namespace: {legacy_modules}"
    )


def test_stage_numbering_matches_the_six_stage_dag():
    root = Path(__file__).resolve().parents[1]
    search_roots = [root / "src", root / "configs", root / "scripts", root / "tools"]
    stale = []
    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".md"}:
                continue
            source = path.read_text()
            if any(f"Stage {number}" in source for number in (7, 8, 9)):
                stale.append(str(path.relative_to(root)))
    assert not stale, f"Stale stage numbering outside the six-stage DAG: {stale}"


def test_evaluation_dry_run_is_read_only(tmp_path):
    from oceanpath.contracts import PipelineStage
    from oceanpath.workflows.evaluation import run_evaluation

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    eval_dir = tmp_path / "eval"
    cfg = OmegaConf.create(
        {
            "dry_run": True,
            "exp_name": "reference",
            "platform": {"project_root": str(tmp_path), "output_root": str(tmp_path)},
            "data": {"name": "blca"},
            "encoder": {"name": "uni_v1"},
            "splits": {"name": "kfold5", "scheme": "kfold", "n_folds": 5},
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
        }
    )

    result = run_evaluation(cfg)

    assert result.stage is PipelineStage.EVALUATE
    assert result.status == "dry_run"
    assert not eval_dir.exists()


def test_export_dry_run_is_read_only(tmp_path):
    from oceanpath.contracts import PipelineStage
    from oceanpath.workflows.export import run_export

    train_dir = tmp_path / "train"
    model_dir = train_dir / "final" / "best_fold"
    model_dir.mkdir(parents=True)
    (model_dir / "model.ckpt").touch()
    artifact_dir = tmp_path / "artifact"
    cfg = OmegaConf.create(
        {
            "dry_run": True,
            "exp_name": "reference",
            "platform": {"project_root": str(tmp_path), "output_root": str(tmp_path)},
            "data": {"name": "blca"},
            "encoder": {"name": "uni_v1"},
            "splits": {"name": "kfold5"},
            "train_dir": str(train_dir),
            "eval_dir": str(train_dir / "eval"),
            "export": {
                "artifact_root": str(tmp_path / "artifacts"),
                "artifact_dir": str(artifact_dir),
                "model_strategy": "best_fold",
                "formats": ["onnx"],
                "skip_validation": False,
            },
        }
    )

    result = run_export(cfg)

    assert result.stage is PipelineStage.EXPORT_MODEL
    assert result.status == "dry_run"
    assert not artifact_dir.exists()


def test_foundation_finalization_runs_only_validation_selected_best_fold(tmp_path, monkeypatch):
    import oceanpath.workflows.finalize as finalize

    cfg = OmegaConf.create({"training": {"final_strategies": ["best_fold"]}})
    monkeypatch.setattr(
        finalize,
        "_save_best_fold",
        lambda *_args: {"strategy": "best_fold", "model_path": "model.ckpt"},
    )
    monkeypatch.setattr(
        finalize,
        "_save_ensemble",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ensemble must be opt-in")),
    )
    monkeypatch.setattr(
        finalize,
        "_run_refit",
        lambda *_args: (_ for _ in ()).throw(AssertionError("refit must be opt-in")),
    )

    results = finalize.finalize_models(cfg, tmp_path, n_folds=5, all_fold_metrics=[])

    assert list(results) == ["best_fold"]


def test_finalization_requires_best_fold_in_foundation_contract(tmp_path):
    from oceanpath.workflows.finalize import finalize_models

    cfg = OmegaConf.create({"training": {"final_strategies": ["ensemble"]}})

    with pytest.raises(ValueError, match="must include best_fold"):
        finalize_models(cfg, tmp_path, n_folds=5, all_fold_metrics=[])
