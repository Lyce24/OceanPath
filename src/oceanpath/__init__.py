"""OceanPath package initialization."""

try:
    import torch
except ImportError:  # the ONNX serving image ships without torch
    torch = None


def _patch_cuda_capture_check() -> None:
    """
    Guard Adam/optimizer CUDA graph capture checks on unsupported CUDA setups.

    Some environments report CUDA as available but fail at
    ``torch.cuda.is_current_stream_capturing()`` with an AcceleratorError.
    Optimizers call this during ``step()`` even for CPU tensors, which can
    crash CPU-only training/tests. In that case, treat capture status as False.
    """
    import torch  # bound here: the module-level name is None in the torch-free image

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

    # A marker attribute on the function object, which pyright does not model.
    _safe_is_current_stream_capturing.__oceanpath_safe__ = True  # pyright: ignore[reportFunctionMemberAccess]
    torch.cuda.is_current_stream_capturing = _safe_is_current_stream_capturing


if torch is not None:
    _patch_cuda_capture_check()
