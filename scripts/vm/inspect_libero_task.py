#!/usr/bin/env python3
"""Print the names needed to make an auditable LIBERO scene-patch JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_root(root: Path) -> None:
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _joint_names(sim: object) -> list[str]:
    model = sim.model  # type: ignore[attr-defined]
    for attribute in ("joint_names", "_joint_names"):
        names = getattr(model, attribute, None)
        if names is not None:
            return [str(name) for name in names if name]
    count = getattr(model, "njnt", 0)
    resolver = getattr(model, "joint_id2name", None)
    return [str(resolver(index)) for index in range(count) if resolver(index)] if resolver else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _add_root(args.openvla_root)
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import get_libero_env  # type: ignore[import-not-found]

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_index)
    env, description = get_libero_env(task, "openvla", resolution=256)
    try:
        observation = env.reset()
        initial_states = suite.get_task_init_states(args.task_index)
        report = {
            "suite_name": args.suite,
            "task_index": args.task_index,
            "task_description": description,
            "initial_state_count": len(initial_states),
            "observation_keys": sorted(observation.keys()),
            "joint_names": _joint_names(env.sim),
            "body_names": [str(name) for name in getattr(env.sim.model, "body_names", []) if name],
        }
    finally:
        env.close()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
