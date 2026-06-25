# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Ahead-of-time build of the fused depthwise CUDA kernel.

Produces a precompiled, multi-arch extension module ``spconv_depthwise_C`` so
end users never JIT-compile at first use. Used by
``tools/build_patched_wheel.py --precompile`` but can also be run directly:

    # build the .pyd/.so into ./_built
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 12.0+PTX" \
        python packaging/setup.py build_ext --build-lib _built

The arch list above covers Turing (RTX 20xx), Ampere (RTX 30xx / A-series),
Ada Lovelace (RTX 40xx, RTX 5000/6000 Ada, RTX 3500 Ada) and Blackwell
(RTX 50xx / RTX PRO 6000; needs CUDA >= 12.8). ``+PTX`` adds forward
compatibility to future archs.
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
CU = HERE.parent / "spconv" / "pytorch" / "csrc" / "depthwise.cu"

# default broad arch list; override with TORCH_CUDA_ARCH_LIST.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5 8.0 8.6 8.9 12.0+PTX")

setup(
    name="spconv_depthwise_C",
    version="0.1.0",
    description="Precompiled fused depthwise conv CUDA kernel for spconv",
    ext_modules=[CUDAExtension("spconv_depthwise_C", [str(CU)])],
    cmdclass={"build_ext": BuildExtension},
)
