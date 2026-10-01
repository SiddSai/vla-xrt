from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_xrt.adapters.mock import DeterministicMockAdapter
from vla_xrt.attacks.joint.alternating import alternating_search
from vla_xrt.artifacts.io import read_jsonl, write_json, write_jsonl
from vla_xrt.core.types import InstructionVariant, ScenePatch, TaskSpec
from vla_xrt.protocol.analyze import summarize
from vla_xrt.protocol.factorial import run_factorial


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_config(config_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir) / f"{config['run_name']}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    tasks = [TaskSpec(task, config["safety"]["predicate"], config["safety"]["threshold"]) for task in config["tasks"]]
    instructions = [InstructionVariant(**item) for item in config["instructions"]]
    scenes = [ScenePatch(**item) for item in config["scenes"]]
    adapter = DeterministicMockAdapter()
    rows = run_factorial(adapter, tasks, config["seeds"], instructions, scenes, config["semantic_distance_max"])
    write_jsonl(output / "episodes.jsonl", [row.to_dict() for row in rows])
    write_json(output / "summary.json", summarize([row.to_dict() for row in rows]))
    write_json(output / "manifest.json", {
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_path": str(config_path.resolve()),
        "adapter": adapter.name,
        "python": sys.version,
        "platform": platform.platform(),
        "result_count": len(rows),
        "fixture_run": True,
    })
    print(output)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    print(json.dumps(summarize(read_jsonl(Path(args.episodes))), indent=2, sort_keys=True))
    return 0


def _search(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    task = TaskSpec(config["tasks"][0], config["safety"]["predicate"], config["safety"]["threshold"])
    instructions = [InstructionVariant(**item) for item in config["instructions"]]
    scenes = [ScenePatch(**item) for item in config["scenes"]]
    result = alternating_search(
        DeterministicMockAdapter(), task, args.seed, instructions, scenes,
        config["semantic_distance_max"], args.rounds,
    )
    print(json.dumps({
        "task": task.id,
        "seed": args.seed,
        "instruction": result.instruction.id,
        "scene": result.scene.id,
        "policy_queries": len(result.observations),
        "observations": [item.__dict__ for item in result.observations],
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="vla-xrt")
    subparsers = parser.add_subparsers(required=True)
    run = subparsers.add_parser("run", help="run the deterministic protocol fixture")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", default="results/runs")
    run.set_defaults(handler=_run)
    analyze = subparsers.add_parser("analyze", help="summarize JSONL episode records")
    analyze.add_argument("episodes")
    analyze.set_defaults(handler=_analyze)
    search = subparsers.add_parser("search", help="run transparent alternating search on a candidate bank")
    search.add_argument("--config", required=True)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--rounds", type=int, default=2)
    search.set_defaults(handler=_search)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
