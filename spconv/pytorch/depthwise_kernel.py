# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loader for the fused sparse depthwise convolution CUDA kernel.

Resolution order (first that works wins, then memoised):

1. ``import spconv_depthwise_C`` -- a *precompiled* extension. This is what
   ships in a wheel built with ``tools/build_patched_wheel.py --precompile``:
   end users pay NO compilation at first use. Built as a multi-arch fatbin so a
   single binary runs on Turing/Ampere/Ada/Blackwell GPUs.
2. JIT compile ``csrc/depthwise.cu`` with ``torch.utils.cpp_extension.load``
   (needs a CUDA toolchain at runtime). Compiled once and cached by torch.
3. Pure-torch fallback (no kernel) -- see ``ops.indice_conv_depthwise``.

The kernel source (``csrc/depthwise.cu``) is shared between the JIT path and
the AOT build (``packaging/setup.py``), so there is a single source of truth.

Env vars:
  SPCONV_DEPTHWISE_DISABLE_CUDA=1   force the pure-torch fallback.
  TORCH_CUDA_ARCH_LIST=...          archs for the JIT build (defaults to the
                                    current GPU only).
"""
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_KERNEL = None
_TRIED = False

_CU_SOURCE = Path(__file__).resolve().parent / "csrc" / "depthwise.cu"

# Broad multi-arch list for AOT builds (see packaging/setup.py). Covers
# Turing (7.5), Ampere (8.0/8.6), Ada Lovelace (8.9) and Blackwell (12.0,
# needs CUDA >= 12.8); +PTX gives forward compatibility to newer archs.
DEFAULT_AOT_ARCH_LIST = "7.5 8.0 8.6 8.9 12.0+PTX"


def get_depthwise_kernel():
    """Return the depthwise CUDA module, or ``None`` to use the torch fallback.

    Thread safe and memoised: resolution is attempted once and the result
    (module or ``None``) cached so callers never retry on the hot path.
    """
    global _KERNEL, _TRIED
    if _TRIED:
        return _KERNEL
    with _LOCK:
        if _TRIED:
            return _KERNEL
        _TRIED = True
        _KERNEL = _resolve_kernel()
        return _KERNEL


def _resolve_kernel():
    if os.environ.get("SPCONV_DEPTHWISE_DISABLE_CUDA", "0") == "1":
        return None
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None

    # 1) precompiled extension shipped in the wheel -> zero compilation.
    try:
        import spconv_depthwise_C  # type: ignore
        return spconv_depthwise_C
    except Exception:
        pass

    # 2) JIT compile from the shared .cu source.
    try:
        if not _CU_SOURCE.exists():
            raise FileNotFoundError(_CU_SOURCE)
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            # compile only for the current GPU (fast build, no warning).
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        from torch.utils.cpp_extension import load
        return load(name="spconv_depthwise_C",
                    sources=[str(_CU_SOURCE)],
                    verbose=False)
    except Exception as e:  # pragma: no cover - depends on toolchain
        import warnings
        warnings.warn(
            "spconv depthwise: fused CUDA kernel unavailable, falling back to "
            f"the (slower) pure-torch path. Reason: {e}")
        return None
