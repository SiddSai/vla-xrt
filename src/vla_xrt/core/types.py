from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Arm(str, Enum):
    CLEAN = "clean"
    INSTRUCTION = "instruction"
    SCENE = "scene"
    JOINT = "joint"


@dataclass(frozen=True)
class InstructionVariant:
    id: str
    text: str
    semantic_distance: float


@dataclass(frozen=True)
class ScenePatch:
    id: str
    kind: str
    distance_cm: float
    feasible: bool


@dataclass(frozen=True)
class TaskSpec:
    id: str
    safety_predicate: str
    safety_threshold: float


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    seed: int
    arm: Arm
    instruction_id: str
    instruction: str
    scene_id: str
    scene_kind: str
    task_success: bool
    feasible: bool
    semantic_preserved: bool
    unsafe: bool
    safety_cost: float
    event_step: int | None
    safety_predicate: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
