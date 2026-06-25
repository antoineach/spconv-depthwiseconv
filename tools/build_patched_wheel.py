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
import glob
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

    # ensure the 'wheel' tool is available
    run([sys.executable, "-m", "pip", "install", "-q", "wheel"])

    # 1) fetch the prebuilt wheel only (never build from source)
    print(f"[1/4] downloading prebuilt wheel for {args.spec}")
    run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
         "--no-deps", args.spec, "-d", str(dl)])
    whls = list(dl.glob("*.whl"))
    if not whls:
        raise SystemExit("no prebuilt wheel downloaded (no cp wheel for this "
                         "python? try another CUDA variant / version)")
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

    # 3) overlay our patched files
    print("[3/4] injecting depthwise files")
    for rel in FILES:
        src = src_pkg / rel
        dst = pkg_dir / "spconv" / rel
        if not src.exists():
            raise SystemExit(f"missing source file: {src}")
        if not dst.parent.exists():
            raise SystemExit(f"target package has no {dst.parent}")
        print(f"      {rel}")
        shutil.copy2(src, dst)

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
