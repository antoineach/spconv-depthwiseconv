#!/usr/bin/env python
# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Overlay the depthwise feature onto an already installed (prebuilt) spconv.

The sparse depthwise convolution is implemented as *pure Python* on top of the
ops that are already in the compiled backend (``spconv.core_cc``). So you do
NOT need a C++/CUDA toolchain (no MSVC) to use it: just install a matching
prebuilt wheel and copy these Python files over it.

Steps
-----
1. Install a prebuilt spconv matching your CUDA, SAME minor version as this
   repo (check ``version.txt``). Example for CUDA 12.0::

       pip install spconv-cu120==2.3.8

   (cpu only:  pip install spconv==2.3.8)

2. From the root of THIS repo, run::

       python tools/install_depthwise_over_prebuilt.py

   It copies the 4 modified files into the installed package:
       spconv/pytorch/{ops.py, functional.py, conv.py, __init__.py}
   A timestamped backup of each overwritten file is kept next to it.

3. Verify::

       python test/verify_depthwise.py

Pass ``--dry-run`` to only print what would happen.
"""
import argparse
import importlib.util
import shutil
import sys
import time
from pathlib import Path

FILES = [
    "pytorch/ops.py",
    "pytorch/functional.py",
    "pytorch/conv.py",
    "pytorch/__init__.py",
]


def find_installed_spconv(repo_root: Path) -> Path:
    """Locate the installed spconv package, ignoring this source checkout."""
    # remove cwd / repo root from path so we don't "find" the source tree.
    repo_root = repo_root.resolve()
    saved = list(sys.path)
    sys.path = [p for p in sys.path
                if p and Path(p).resolve() != repo_root and Path(p).resolve() != Path.cwd().resolve()]
    try:
        spec = importlib.util.find_spec("spconv")
    finally:
        sys.path = saved
    if spec is None or not spec.origin:
        raise SystemExit(
            "Could not find an installed spconv. Install a prebuilt wheel first, "
            "e.g.  pip install spconv-cu120==<version in version.txt>")
    installed = Path(spec.origin).parent
    if installed.resolve() == (repo_root / "spconv").resolve():
        raise SystemExit(
            "find_spec resolved to this source checkout, not an installed wheel.\n"
            "Run this script from a directory that is NOT the repo root, or make "
            "sure a prebuilt spconv is pip-installed.")
    # sanity: prebuilt must ship the compiled backend.
    if not (installed / "core_cc").exists() and not list(installed.glob("core_cc*")):
        print("WARNING: installed spconv has no 'core_cc' compiled backend; "
              "is this really a prebuilt wheel?", file=sys.stderr)
    return installed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_pkg = repo_root / "spconv"

    installed = find_installed_spconv(repo_root)
    print(f"source (this repo):  {src_pkg}")
    print(f"installed spconv:    {installed}")

    # version sanity check
    src_ver = (repo_root / "version.txt").read_text().strip()
    inst_ver_file = installed / "version.txt"
    if inst_ver_file.exists():
        inst_ver = inst_ver_file.read_text().strip()
        if inst_ver != src_ver:
            print(f"WARNING: version mismatch repo={src_ver} installed={inst_ver}. "
                  "Overlaying across different versions may break things. "
                  "Prefer matching versions.", file=sys.stderr)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for rel in FILES:
        src = src_pkg / rel
        dst = installed / rel
        if not src.exists():
            raise SystemExit(f"missing source file: {src}")
        print(f"  {dst}  <-  {rel}")
        if args.dry_run:
            continue
        if dst.exists():
            shutil.copy2(dst, dst.with_suffix(dst.suffix + f".bak-{stamp}"))
        shutil.copy2(src, dst)

    if args.dry_run:
        print("\n(dry run, nothing copied)")
    else:
        print("\nDone. Verify with:  python test/verify_depthwise.py")


if __name__ == "__main__":
    main()
