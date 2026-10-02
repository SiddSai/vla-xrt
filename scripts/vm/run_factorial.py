#!/usr/bin/env python3
"""Sequential, restartable runner for one frozen VLA-XRT LIBERO case file."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="LIBERO initial-state indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    instructions = [item for item in case["instructions"] if item.get("human_approved")]
    scenes = [item for item in case["scenes"] if item.get("feasible")]
    if not any(item["id"] == "clean" for item in instructions + scenes):
        raise ValueError("Case must include approved clean instruction and feasible clean scene")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode = Path(__file__).with_name("run_openvla_libero_episode.py")
    for seed, instruction, scene in itertools.product(args.seeds, instructions, scenes):
        output = args.output_dir / f"seed-{seed}__instruction-{instruction['id']}__scene-{scene['id']}.json"
        if output.exists():
            print(f"skip existing {output}")
            continue
        command = [
            sys.executable, str(episode), "--openvla-root", str(args.openvla_root), "--checkpoint", args.checkpoint,
            "--case", str(args.case), "--instruction-id", instruction["id"], "--scene-id", scene["id"],
            "--seed", str(seed), "--output", str(output),
        ]
        if args.center_crop:
            command.append("--center-crop")
        if args.max_steps:
            command.extend(("--max-steps", str(args.max_steps)))
        print("running", " ".join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
