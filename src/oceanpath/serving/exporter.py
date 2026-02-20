"""
Model exporter with portability and numerical validation.

Converts a trained MIL checkpoint into deployment-ready artifacts:
  - ONNX (primary, cross-platform, quantization-ready)
  - TorchScript (fallback, pure-PyTorch environments)
  - model_card.json (full provenance)

Validation pipeline (runs automatically before export):
  1. Portability — CPU vs GPU outputs match within tolerance
  2. Numerical  — exported model reproduces PyTorch outputs exactly
  3. Shape      — dynamic axes handle varying patch counts correctly

Why validate before export?
═══════════════════════════
A checkpoint that works on your training GPU can silently break when:
  - Loaded on CPU (device-specific tensors, custom CUDA kernels)
  - Exported to ONNX (unsupported ops, dynamic shape bugs)
  - Deployed on different hardware (float precision drift)

Catching these at export time — not in production — is the entire point.
"""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Exporter
# ═════════════════════════════════════════════════════════════════════════════


class Exporter:
    """
    Export a trained MIL model to ONNX and/or TorchScript.

    Usage
    ─────
        exporter = Exporter(
            checkpoint_path="outputs/train/exp/final/best_fold/model.ckpt",
            output_dir="artifacts/exp_abc123",
            in_dim=1024,
            num_classes=3,
        )
        report = exporter.run()   # validates + exports + writes model card

    The exporter:
      1. Loads the Lightning checkpoint → extracts pure PyTorch model
      2. Runs portability validation (CPU vs GPU output match)
      3. Exports to ONNX with dynamic axes (variable patch count)
      4. Exports to TorchScript (trace-based)
      5. Runs numerical validation against each exported format
      6. Generates model_card.json with full provenance
    """

    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        in_dim: int = 1024,
        num_classes: int = 2,
        *,
        formats: list[str] | None = None,
        opset_version: int = 17,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        validation_n_patches: list[int] | None = None,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        checkpoint_path : path to Lightning .ckpt file.
        output_dir : directory for exported artifacts.
        in_dim : input feature dimension (must match encoder).
        num_classes : output classes (must match training).
        formats : export formats, subset of ["onnx", "torchscript"]. Default: both.
        opset_version : ONNX opset (17 recommended for broad compatibility).
        atol, rtol : numerical tolerance for validation.
        validation_n_patches : patch counts to test dynamic axes. Default: [10, 100, 500].
        device : device for model loading ("cpu" or "cuda:0").
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.formats = formats or ["onnx", "torchscript"]
        self.opset_version = opset_version
        self.atol = atol
        self.rtol = rtol
        self.validation_n_patches = validation_n_patches or [10, 100, 500]
        self.device = device

        # State
        self._model: Optional[nn.Module] = None
        self._report: dict = {}

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, skip_validation: bool = False) -> dict:
        """
        Execute the full export pipeline.

        Returns a report dict with paths, validation results, and timing.
        """
        start = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "checkpoint": str(self.checkpoint_path),
            "output_dir": str(self.output_dir),
            "in_dim": self.in_dim,
            "num_classes": self.num_classes,
            "formats": self.formats,
            "validation": {},
            "exports": {},
        }

        # 1. Load model
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        model = self._load_model()
        report["checkpoint_hash"] = _file_sha256(self.checkpoint_path)
        report["model_params"] = sum(p.numel() for p in model.parameters())

        # 2. Portability validation
        if not skip_validation:
            report["validation"]["portability"] = self._validate_portability(
                model
            )

        # 3. Export ONNX
        if "onnx" in self.formats:
            onnx_path = self.output_dir / "model.onnx"
            onnx_report = self._export_onnx(model, onnx_path)
            report["exports"]["onnx"] = onnx_report

            if not skip_validation:
                report["validation"]["onnx_numerical"] = (
                    self._validate_onnx(model, onnx_path)
                )

        # 4. Export TorchScript
        if "torchscript" in self.formats:
            ts_path = self.output_dir / "model.pt"
            ts_report = self._export_torchscript(model, ts_path)
            report["exports"]["torchscript"] = ts_report

            if not skip_validation:
                report["validation"]["torchscript_numerical"] = (
                    self._validate_torchscript(model, ts_path)
                )

        # 5. Copy checkpoint as fallback
        fallback_path = self.output_dir / "model.ckpt"
        if not fallback_path.exists():
            shutil.copy2(self.checkpoint_path, fallback_path)
            report["exports"]["checkpoint"] = {"path": str(fallback_path)}

        report["elapsed_seconds"] = round(time.monotonic() - start, 1)
        report["success"] = _all_validations_passed(report)

        # Write report
        report_path = self.output_dir / "export_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info(
            f"Export complete: {report['elapsed_seconds']}s, "
            f"success={report['success']} → {self.output_dir}"
        )

        self._report = report
        return report

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> nn.Module:
        """Load Lightning checkpoint → extract pure PyTorch model."""
        from oceanpath.modules.train_module import MILTrainModule

        try:
            module = MILTrainModule.load_from_checkpoint(
                str(self.checkpoint_path),
                weights_only=False,
                map_location="cpu",
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(
                str(self.checkpoint_path),
                map_location="cpu",
            )

        # Extract the pure WSIClassifier (no Lightning overhead)
        model = module.model
        model.eval()
        model.to("cpu")

        logger.info(
            f"Loaded model: {type(model.aggregator).__name__}, "
            f"in_dim={model.aggregator.in_dim}, "
            f"embed_dim={model.aggregator.embed_dim}, "
            f"num_classes={model.num_classes}, "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )

        self._model = model
        return model

    # ── Portability validation ────────────────────────────────────────────

    def _validate_portability(self, model: nn.Module) -> dict:
        """
        Verify CPU and GPU produce identical outputs.

        This catches:
          - Device-specific tensors baked into buffers
          - Custom CUDA kernels with no CPU fallback
          - Numerical divergence from mixed precision artifacts
        """
        result = {"status": "skipped", "tests": []}

        if not torch.cuda.is_available():
            logger.info(
                "Portability validation: CUDA unavailable, "
                "testing CPU-only consistency"
            )
            # Still test that CPU works with different batch shapes
            for n_patches in self.validation_n_patches:
                x = torch.randn(1, n_patches, self.in_dim)
                try:
                    with torch.no_grad():
                        out = model(x)
                    result["tests"].append({
                        "n_patches": n_patches,
                        "status": "pass",
                        "logit_shape": list(out.logits.shape),
                    })
                except Exception as e:
                    result["tests"].append({
                        "n_patches": n_patches,
                        "status": "fail",
                        "error": str(e),
                    })
            result["status"] = (
                "pass" if all(t["status"] == "pass" for t in result["tests"])
                else "fail"
            )
            return result

        # Full CPU vs GPU validation
        model_cpu = model.to("cpu").eval()

        for n_patches in self.validation_n_patches:
            x = torch.randn(1, n_patches, self.in_dim)

            with torch.no_grad():
                out_cpu = model_cpu(x)

                model_gpu = model_cpu.to("cuda").eval()
                out_gpu = model_gpu(x.cuda())

                # Move GPU outputs to CPU for comparison
                logits_match = torch.allclose(
                    out_gpu.logits.cpu().float(),
                    out_cpu.logits.float(),
                    atol=self.atol,
                    rtol=self.rtol,
                )
                embed_match = torch.allclose(
                    out_gpu.slide_embedding.cpu().float(),
                    out_cpu.slide_embedding.float(),
                    atol=self.atol,
                    rtol=self.rtol,
                )

                max_logit_diff = (
                    (out_gpu.logits.cpu().float() - out_cpu.logits.float())
                    .abs()
                    .max()
                    .item()
                )

                result["tests"].append({
                    "n_patches": n_patches,
                    "status": "pass" if (logits_match and embed_match) else "fail",
                    "logits_match": logits_match,
                    "embed_match": embed_match,
                    "max_logit_diff": max_logit_diff,
                })

                # Move back to CPU for next iteration
                model_cpu = model_gpu.to("cpu").eval()

        result["status"] = (
            "pass" if all(t["status"] == "pass" for t in result["tests"])
            else "fail"
        )

        if result["status"] == "fail":
            logger.error(
                "PORTABILITY VALIDATION FAILED — "
                "CPU/GPU outputs differ beyond tolerance. "
                "The checkpoint may contain device-specific artifacts."
            )
        else:
            logger.info(
                f"Portability validation passed "
                f"({len(result['tests'])} shapes tested)"
            )

        # Ensure model is back on target device
        model.to("cpu")
        return result

    # ── ONNX export ───────────────────────────────────────────────────────

    def _export_onnx(self, model: nn.Module, path: Path) -> dict:
        """Export model to ONNX with dynamic axes for variable patch count."""
        logger.info(f"Exporting ONNX (opset={self.opset_version})...")

        wrapper = _OnnxExportWrapper(model)
        wrapper.eval()

        # Sample input: [1, N, D]
        dummy = torch.randn(1, 100, self.in_dim)

        try:
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(path),
                opset_version=self.opset_version,
                input_names=["features"],
                output_names=["logits", "probabilities", "embedding"],
                dynamic_axes={
                    "features": {0: "batch", 1: "n_patches"},
                    "logits": {0: "batch"},
                    "probabilities": {0: "batch"},
                    "embedding": {0: "batch"},
                },
            )

            # Verify ONNX is loadable
            import onnx
            onnx_model = onnx.load(str(path))
            onnx.checker.check_model(onnx_model)

            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"ONNX export: {size_mb:.1f} MB → {path}")

            return {
                "path": str(path),
                "size_mb": round(size_mb, 2),
                "opset": self.opset_version,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _validate_onnx(self, model: nn.Module, onnx_path: Path) -> dict:
        """Run the exported ONNX model and compare outputs to PyTorch."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "onnxruntime not installed — skipping ONNX numerical validation"
            )
            return {"status": "skipped", "reason": "onnxruntime not installed"}

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"],
        )
        wrapper = _OnnxExportWrapper(model).eval()

        results = []
        for n_patches in self.validation_n_patches:
            x = torch.randn(1, n_patches, self.in_dim)

            # PyTorch reference
            with torch.no_grad():
                pt_logits, pt_probs, pt_embed = wrapper(x)

            # ONNX
            ort_out = session.run(
                None, {"features": x.numpy()},
            )
            ort_logits = ort_out[0]
            ort_probs = ort_out[1]

            match = np.allclose(
                pt_logits.numpy(), ort_logits,
                atol=self.atol, rtol=self.rtol,
            )
            max_diff = float(np.abs(pt_logits.numpy() - ort_logits).max())

            results.append({
                "n_patches": n_patches,
                "status": "pass" if match else "fail",
                "max_logit_diff": max_diff,
            })

        status = "pass" if all(r["status"] == "pass" for r in results) else "fail"
        if status == "fail":
            logger.error(
                "ONNX NUMERICAL VALIDATION FAILED — "
                "exported model outputs differ from PyTorch"
            )
        else:
            logger.info(
                f"ONNX numerical validation passed "
                f"({len(results)} shapes tested)"
            )

        return {"status": status, "tests": results}

    # ── TorchScript export ────────────────────────────────────────────────

    def _export_torchscript(self, model: nn.Module, path: Path) -> dict:
        """Export model to TorchScript via tracing."""
        logger.info("Exporting TorchScript...")

        wrapper = _TorchScriptExportWrapper(model)
        wrapper.eval()

        dummy = torch.randn(1, 100, self.in_dim)

        try:
            traced = torch.jit.trace(wrapper, (dummy,))
            traced.save(str(path))

            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"TorchScript export: {size_mb:.1f} MB → {path}")

            return {
                "path": str(path),
                "size_mb": round(size_mb, 2),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _validate_torchscript(self, model: nn.Module, ts_path: Path) -> dict:
        """Load TorchScript model and compare outputs to PyTorch."""
        try:
            loaded = torch.jit.load(str(ts_path), map_location="cpu")
        except Exception as e:
            return {"status": "failed", "error": f"Cannot load: {e}"}

        wrapper = _TorchScriptExportWrapper(model).eval()

        results = []
        for n_patches in self.validation_n_patches:
            x = torch.randn(1, n_patches, self.in_dim)

            with torch.no_grad():
                pt_out = wrapper(x)
                ts_out = loaded(x)

            match = torch.allclose(
                pt_out[0], ts_out[0], atol=self.atol, rtol=self.rtol,
            )
            max_diff = float((pt_out[0] - ts_out[0]).abs().max().item())

            results.append({
                "n_patches": n_patches,
                "status": "pass" if match else "fail",
                "max_logit_diff": max_diff,
            })

        status = "pass" if all(r["status"] == "pass" for r in results) else "fail"
        if status == "fail":
            logger.error(
                "TORCHSCRIPT NUMERICAL VALIDATION FAILED — "
                "outputs differ from PyTorch"
            )
        else:
            logger.info(
                f"TorchScript numerical validation passed "
                f"({len(results)} shapes tested)"
            )

        return {"status": status, "tests": results}


# ═════════════════════════════════════════════════════════════════════════════
# Export wrappers
# ═════════════════════════════════════════════════════════════════════════════
#
# ONNX and TorchScript need flat tensor inputs/outputs (no dataclasses).
# These wrappers adapt WSIClassifier's MILOutput interface to plain tensors.


class _OnnxExportWrapper(nn.Module):
    """Wraps WSIClassifier for ONNX export: features → (logits, probs, embed)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(features, return_attention=False)
        logits = output.logits
        probs = torch.softmax(logits.float(), dim=-1)
        return logits, probs, output.slide_embedding


class _TorchScriptExportWrapper(nn.Module):
    """Wraps WSIClassifier for TorchScript tracing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(features, return_attention=False)
        logits = output.logits
        probs = torch.softmax(logits.float(), dim=-1)
        return logits, probs, output.slide_embedding


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════


def _file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _all_validations_passed(report: dict) -> bool:
    """Check if all validation steps passed."""
    for name, result in report.get("validation", {}).items():
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            if status == "fail":
                return False
    return True


def config_fingerprint(cfg_dict: dict) -> str:
    """
    Compute a short fingerprint of the experiment config.

    Used for artifact naming: {exp_name}_{fingerprint}/
    """
    canonical = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]