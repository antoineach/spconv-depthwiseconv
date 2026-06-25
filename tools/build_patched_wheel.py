#!/usr/bin/env python
# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Build a drop-in spconv wheel that bundles the depthwise feature.

Instead of recompiling spconv from source (heavy: needs cumm + CUDA + a C++
toolchain), this repacks an existing PREBUILT spconv wheel and swaps in the few
patched pure-Python files (incl. the JIT depthwise CUDA kernel module). The
result is a single ``.whl`` that anyone can ``pip install`` and that behaves
exactly like the upstream wheel plus the depthwise layers.

    python tools/build_patched_wheel.py --spec spconv-cu120==2.3.8

Output: ``dist/spconv_cuXXX-<ver>-<build>-...whl``. Upload it to a GitHub
Release so others can:

    pip install https://github.com/<you>/<repo>/releases/download/<tag>/<file>.whl

Notes
-----
* No C++/CUDA toolchain is needed to BUILD this wheel (it only copies files).
* The depthwise CUDA kernel still JIT-compiles on first use on the end-user
  machine (or falls back to pure-torch). It is not precompiled into the wheel
  because that would tie it to one exact python/torch/CUDA ABI.
* Pick the CUDA variant + version that has a prebuilt wheel for the target
  Python (e.g. cu120/cu124/cu126; cp39-cp313 depending on release).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

FILES = [
    "pytorch/ops.py",
    "pytorch/functional.py",
    "pytorch/conv.py",
    "pytorch/__init__.py",
    "pytorch/depthwise_kernel.py",
    "pytorch/csrc/depthwise.cu",
]


def run(cmd):
    print("  $", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="spconv-cu120==2.3.8",
                    help="prebuilt spconv pip spec to repack")
    ap.add_argument("--outdir", default="dist")
    ap.add_argument("--build-number", default="1",
                    help="wheel build tag to distinguish from upstream")
    ap.add_argument("--workdir", default="build_wheel")
    ap.add_argument("--precompile", action="store_true",
                    help="compile the fused CUDA kernel and bundle it in the "
                    "wheel so end users never JIT-compile (needs CUDA + a C++ "
                    "toolchain on THIS machine; the wheel becomes specific to "
                    "this python/torch/OS)")
    ap.add_argument("--arch", default="7.5 8.0 8.6 8.9 12.0+PTX",
                    help="TORCH_CUDA_ARCH_LIST for --precompile (Turing..Blackwell)")
    ap.add_argument("--wheel", default=None,
                    help="path to a LOCAL prebuilt spconv .whl to repack "
                    "(fully offline; skips the pip download of --spec)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_pkg = repo_root / "spconv"
    work = (repo_root / args.workdir).resolve()
    dl = work / "download"
    unpacked = work / "unpacked"
    outdir = (repo_root / args.outdir).resolve()
    for d in (dl, unpacked):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # ensure the 'wheel' tool is available (offline if already installed)
    try:
        import wheel  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "wheel"])

    # 1) obtain the prebuilt wheel (local file => fully offline, else download)
    if args.wheel:
        whl = Path(args.wheel).expanduser().resolve()
        if not whl.is_file():
            raise SystemExit(f"--wheel not found: {whl}")
        print(f"[1/4] using local wheel {whl.name} (offline)")
    else:
        print(f"[1/4] downloading prebuilt wheel for {args.spec}")
        run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
             "--no-deps", args.spec, "-d", str(dl)])
        whls = list(dl.glob("*.whl"))
        if not whls:
            raise SystemExit("no prebuilt wheel downloaded (no cp wheel for "
                             "this python? try another CUDA variant / version)")
        whl = whls[0]
        print(f"      got {whl.name}")

    # 2) unpack
    print("[2/4] unpacking")
    run([sys.executable, "-m", "wheel", "unpack", str(whl), "-d",
         str(unpacked)])
    pkg_dirs = [p for p in unpacked.iterdir() if p.is_dir()]
    if len(pkg_dirs) != 1:
        raise SystemExit(f"unexpected unpack layout: {pkg_dirs}")
    pkg_dir = pkg_dirs[0]

    # sanity: the base wheel must be a real PREBUILT (ships the compiled
    # backend `spconv/core_cc`). A `py3-none-any` source wheel JIT-compiles
    # core_cc at runtime and is useless on a machine without a toolchain.
    core_cc = list((pkg_dir / "spconv").glob("core_cc*"))
    if not core_cc:
        print("WARNING: base wheel has no 'spconv/core_cc' compiled backend.\n"
              "         This looks like a source/py3-none-any wheel: the result"
              " will\n         NOT work on a machine without CUDA+ninja (spconv"
              " itself will\n         fail to import). Use a real prebuilt wheel"
              " instead, e.g.\n         pip download --only-binary=:all: "
              "--no-deps spconv-cu120==2.3.8\n", file=sys.stderr)

    # 3) overlay our patched files
    print("[3/4] injecting depthwise files")
    for rel in FILES:
        src = src_pkg / rel
        dst = pkg_dir / "spconv" / rel
        if not src.exists():
            raise SystemExit(f"missing source file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"      {rel}")
        shutil.copy2(src, dst)

    # 3b) optionally compile the fused kernel and bundle it (no JIT for users)
    if args.precompile:
        print("[3b] precompiling fused CUDA kernel (arch:", args.arch, ")")
        # Build from a SHORT directory with a relative source filename so the
        # mirrored build/temp path stays under Windows MAX_PATH (260 chars).
        short = Path.home() / ".spconv_dw_build"
        if short.exists():
            shutil.rmtree(short, ignore_errors=True)
        short.mkdir(parents=True)
        shutil.copy2(src_pkg / "pytorch" / "csrc" / "depthwise.cu",
                     short / "depthwise.cu")
        built_lib = short / "lib"
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST=args.arch,
                   SPCONV_DW_CU="depthwise.cu")
        setup_py = repo_root / "packaging" / "setup.py"
        subprocess.check_call(
            [sys.executable, str(setup_py), "build_ext",
             "--build-temp", str(short / "t"), "--build-lib", str(built_lib)],
            cwd=str(short), env=env)
        exts = [p for p in built_lib.iterdir()
                if p.name.startswith("spconv_depthwise_C")
                and p.suffix in (".pyd", ".so")] if built_lib.exists() else []
        if not exts:
            raise SystemExit(f"no compiled extension found in {built_lib}")
        for ext in exts:
            print(f"      bundling {ext.name}")
            shutil.copy2(ext, pkg_dir / ext.name)  # top-level module in wheel

    # 4) repack (wheel pack regenerates RECORD hashes)
    print("[4/4] repacking")
    run([sys.executable, "-m", "wheel", "pack", str(pkg_dir), "-d",
         str(outdir), "--build-number", args.build_number])

    built = sorted(outdir.glob("*.whl"), key=lambda p: p.stat().st_mtime)[-1]
    print("\nDone:", built)
    print("\nInstall with:")
    print(f"  pip install --force-reinstall --no-deps \"{built}\"")
    print("\nOr publish it to a GitHub Release and:")
    print("  pip install https://github.com/<you>/<repo>/releases/download/"
          f"<tag>/{built.name}")


if __name__ == "__main__":
    main()
