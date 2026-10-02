#!/usr/bin/env python3
"""Fail fast before downloading a checkpoint or spending GPU time.

This script intentionally has no VLA-XRT dependency.  It verifies the active
environment into which the official OpenVLA and LIBERO checkouts were installed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def version_of(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:  # report import diagnostics rather than conceal them
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gpu = {"available": False}
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, capture_output=True, check=False,
        )
        gpu = {"available": completed.returncode == 0, "details": completed.stdout.strip() or completed.stderr.strip()}
    modules = {name: version_of(name) for name in ("torch", "mujoco", "robosuite", "libero", "transformers")}
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
        "modules": modules,
        "openvla_root": str(args.openvla_root.resolve()),
        "openvla_revision": git_revision(args.openvla_root),
        "openvla_evaluator_present": (args.openvla_root / "experiments/robot/libero/run_libero_eval.py").is_file(),
        "libero_root": str(args.libero_root.resolve()),
        "libero_revision": git_revision(args.libero_root),
    }
    required = [
        platform.system() == "Linux",
        report["mujoco_gl"] == "egl",
        gpu["available"],
        report["openvla_evaluator_present"],
        all(value["available"] for value in modules.values()),
    ]
    report["ready"] = all(required)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
