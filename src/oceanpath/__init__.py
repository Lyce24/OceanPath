"""OceanPath package initialization."""

import torch


def _patch_cuda_capture_check() -> None:
    """
    Guard Adam/optimizer CUDA graph capture checks on unsupported CUDA setups.

    Some environments report CUDA as available but fail at
    ``torch.cuda.is_current_stream_capturing()`` with an AcceleratorError.
    Optimizers call this during ``step()`` even for CPU tensors, which can
    crash CPU-only training/tests. In that case, treat capture status as False.
    """

    fn = getattr(torch.cuda, "is_current_stream_capturing", None)
    if fn is None or getattr(fn, "__oceanpath_safe__", False):
        return

    def _safe_is_current_stream_capturing() -> bool:
        try:
            return bool(fn())
        except torch.AcceleratorError:
            return False
        except RuntimeError:
            return False

    _safe_is_current_stream_capturing.__oceanpath_safe__ = True  # type: ignore[attr-defined]
    torch.cuda.is_current_stream_capturing = _safe_is_current_stream_capturing


_patch_cuda_capture_check()
