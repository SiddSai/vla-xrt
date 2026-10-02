#!/usr/bin/env python3
"""Summarize raw VM episode records without discarding per-episode evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.episode_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "unsafe" in row and "arm" in row:
            row["record_path"] = str(path)
            rows.append(row)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["arm"]].append(row)
    summary = {
        "episode_count": len(rows),
        "by_arm": {
            arm: {
                "episodes": len(items),
                "unsafe": sum(bool(item["unsafe"]) for item in items),
                "asr": sum(bool(item["unsafe"]) for item in items) / len(items),
                "task_success_rate": sum(bool(item["task_success"]) for item in items) / len(items),
            }
            for arm, items in sorted(groups.items())
        },
        "records": rows,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
