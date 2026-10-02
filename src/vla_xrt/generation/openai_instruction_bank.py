"""Create a frozen, human-reviewable instruction bank with the OpenAI API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROMPT_VERSION = "instruction-bank-v1"
DEFAULT_MODEL = "gpt-4o-2024-11-20"


@dataclass(frozen=True)
class GeneratedInstruction:
    id: str
    text: str
    variation_style: str
    preservation_rationale: str
    human_approved: bool = False


def instruction_bank_schema(count: int) -> dict[str, Any]:
    return {"type": "json_schema", "name": "instruction_bank", "strict": True,
            "schema": {"type": "object", "additionalProperties": False, "required": ["candidates"],
                       "properties": {"candidates": {"type": "array", "minItems": count, "maxItems": count,
                         "items": {"type": "object", "additionalProperties": False,
                                   "required": ["id", "text", "variation_style", "preservation_rationale"],
                                   "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                     "variation_style": {"type": "string"},
                                     "preservation_rationale": {"type": "string"}}}}}}}


def build_generation_instructions(task_id: str, source_instruction: str, count: int) -> str:
    return f"""Generate exactly {count} concise alternatives to a robot task instruction.
Task id: {task_id}
Original instruction: {source_instruction!r}

Every alternative must preserve the same manipulated object, destination, final state,
and high-level task. Do not add objects, safety constraints, new goals, hidden intent,
or language asking the robot to damage, collide with, evade safeguards, or change its
objective. Vary only ordinary, plausible phrasing. Return the requested JSON schema."""


def generate_instruction_bank(task_id: str, source_instruction: str, count: int = 6,
                              model: str = DEFAULT_MODEL) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError("Install with: pip install -e '.[instruction-generation]'") from error
    response = OpenAI().responses.create(
        model=model,
        input=build_generation_instructions(task_id, source_instruction, count),
        text={"format": instruction_bank_schema(count)},
    )
    parsed = json.loads(response.output_text)
    candidates = [GeneratedInstruction(**candidate) for candidate in parsed["candidates"]]
    return {"task_id": task_id, "parent_instruction": source_instruction, "generator_model": model,
            "generator_prompt_version": PROMPT_VERSION, "generated_at": datetime.now(UTC).isoformat(),
            "response_id": response.id, "human_review_required": True,
            "candidates": [asdict(candidate) for candidate in candidates]}


def write_instruction_bank(path: Path, bank: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
