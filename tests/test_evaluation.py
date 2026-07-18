"""Contract tests for the basic evaluation domain shipped on main."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from oceanpath.eval.core import (
    aggregate_to_patient_level,
    compute_calibration,
    compute_metrics,
    compute_operating_points,
    extract_probs_and_labels,
)


def test_oof_evaluation_fails_when_training_predictions_are_missing(tmp_path):
    from oceanpath.workflows.evaluation import evaluate_oof

    with pytest.raises(FileNotFoundError, match="Training must complete"):
        evaluate_oof(tmp_path / "train", tmp_path / "eval")


def test_binary_metrics_contract():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(labels, probabilities)

    assert metrics.n_samples == 4
    assert metrics.n_classes == 2
    assert metrics.accuracy == 1.0
    assert metrics.auroc_macro == 1.0


def test_multiclass_probabilities_are_renormalized():
    frame = pd.DataFrame(
        {
            "label": [0, 1],
            "prob_0": [0.9, 0.2],
            "prob_1": [0.2, 0.9],
        }
    )

    labels, probabilities = extract_probs_and_labels(frame)

    np.testing.assert_array_equal(labels, np.array([0, 1]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_patient_aggregation_averages_slide_probabilities():
    frame = pd.DataFrame(
        {
            "slide_id": ["a", "b", "c"],
            "patient_id": ["p1", "p1", "p2"],
            "label": [1, 1, 0],
            "prob_0": [0.2, 0.4, 0.9],
            "prob_1": [0.8, 0.6, 0.1],
        }
    )

    patients = aggregate_to_patient_level(frame)
    patient_one = patients.loc[patients["patient_id"] == "p1"].iloc[0]

    assert len(patients) == 2
    assert patient_one["n_slides"] == 2
    assert patient_one["prob_1"] == 0.7


def test_calibration_contract_is_serializable():
    result = compute_calibration(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        n_bins=5,
    )

    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["brier"] <= 1.0
    assert len(result["bins"]) == 5


def test_binary_operating_points_include_youden_threshold():
    result = compute_operating_points(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert "youdens_j" in result
    assert 0.0 <= result["youdens_j"]["threshold"] <= 1.0
    assert result["confusion_at_youden"] == [[2, 0], [0, 2]]


def _evaluation_workflow_config(tmp_path: Path):
    return OmegaConf.create(
        {
            "exp_name": "reference",
            "platform": {
                "project_root": str(tmp_path),
                "output_root": str(tmp_path / "outputs"),
                "splits_root": str(tmp_path / "splits"),
                "num_workers": 0,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
            },
            "data": {
                "name": "reference",
                "mmap_dir": str(tmp_path / "mmap"),
                "csv_path": str(tmp_path / "manifest.csv"),
                "filename_column": "filename",
                "label_columns": ["label"],
            },
            "encoder": {"name": "uni_v1"},
            "splits": {
                "name": "kfold5",
                "scheme": "kfold",
                "output_dir": str(tmp_path / "splits"),
            },
            "training": {"batch_size": 1, "max_instances": 100},
        }
    )


def test_final_model_evaluation_is_skipped_without_held_out_test(tmp_path, monkeypatch):
    import oceanpath.eval.inference as inference
    import oceanpath.workflows.evaluation as evaluation

    train_dir = tmp_path / "train"
    trained_model_dir = train_dir / "final" / "best_fold"
    trained_model_dir.mkdir(parents=True)
    (trained_model_dir / "info.json").write_text('{"strategy": "best_fold"}')
    pd.DataFrame(
        {
            "slide_id": ["oof_a", "oof_b"],
            "label": [0, 1],
            "prob_0": [0.9, 0.1],
            "prob_1": [0.1, 0.9],
        }
    ).to_parquet(train_dir / "oof_predictions.parquet", index=False)

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "recommended_model.json").write_text('{"recommended_model": "best_fold"}')
    (eval_dir / "model_comparison.json").write_text("{}")
    stale_final_dir = eval_dir / "final_models" / "best_fold"
    stale_final_dir.mkdir(parents=True)
    (stale_final_dir / "performance.json").write_text("{}")

    monkeypatch.setattr(inference, "get_test_slide_ids", lambda *args, **kwargs: [])

    def _unexpected_inference(*args, **kwargs):
        raise AssertionError("OOF predictions must not be reused as final-model test predictions")

    monkeypatch.setattr(inference, "run_inference", _unexpected_inference)

    def _unexpected_final_evaluation(**kwargs):
        raise AssertionError("Final models must not be scored with OOF predictions")

    monkeypatch.setattr(evaluation, "evaluate_final_model", _unexpected_final_evaluation)

    results = evaluation.evaluate_final_models(
        train_dir=train_dir,
        eval_dir=eval_dir,
        cfg=_evaluation_workflow_config(tmp_path),
    )

    scope = json.loads((eval_dir / "evaluation_scope.json").read_text())
    assert results == {}
    assert scope["mode"] == "oof_only"
    assert scope["held_out_test"] is False
    assert not (eval_dir / "recommended_model.json").exists()
    assert not (eval_dir / "model_comparison.json").exists()
    assert not (eval_dir / "final_models").exists()


def test_final_models_still_use_real_held_out_test_ids(tmp_path, monkeypatch):
    import oceanpath.eval.inference as inference
    import oceanpath.workflows.evaluation as evaluation

    train_dir = tmp_path / "train"
    model_dir = train_dir / "final" / "best_fold"
    model_dir.mkdir(parents=True)
    (model_dir / "info.json").write_text('{"strategy": "best_fold"}')
    eval_dir = tmp_path / "eval"
    captured = {}

    monkeypatch.setattr(
        inference,
        "get_test_slide_ids",
        lambda *args, **kwargs: ["held_out_a", "held_out_b"],
    )

    def _fake_inference(**kwargs):
        captured["test_slide_ids"] = kwargs["test_slide_ids"]
        return pd.DataFrame(
            {
                "slide_id": ["held_out_a", "held_out_b"],
                "label": [0, 1],
                "prob_0": [0.8, 0.2],
                "prob_1": [0.2, 0.8],
            }
        )

    monkeypatch.setattr(inference, "run_inference", _fake_inference)
    monkeypatch.setattr(
        evaluation,
        "evaluate_final_model",
        lambda **kwargs: {"performance": {}, "n_predictions": len(kwargs["preds_df"])},
    )

    results = evaluation.evaluate_final_models(
        train_dir=train_dir,
        eval_dir=eval_dir,
        cfg=_evaluation_workflow_config(tmp_path),
    )

    scope = json.loads((eval_dir / "evaluation_scope.json").read_text())
    assert captured["test_slide_ids"] == ["held_out_a", "held_out_b"]
    assert results["best_fold"]["n_predictions"] == 2
    assert scope["mode"] == "held_out_test"
    assert scope["n_test_slides"] == 2
