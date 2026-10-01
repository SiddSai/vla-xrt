from __future__ import annotations

from itertools import product
from typing import Iterable

from vla_xrt.adapters.base import EnvironmentAdapter
from vla_xrt.core.types import Arm, EpisodeResult, InstructionVariant, ScenePatch, TaskSpec
from vla_xrt.oracles.gates import instruction_preserves_task, is_unsafe, scene_is_feasible


def classify_arm(instruction: InstructionVariant, scene: ScenePatch) -> Arm:
    if instruction.id == "clean" and scene.id == "clean":
        return Arm.CLEAN
    if scene.id == "clean":
        return Arm.INSTRUCTION
    if instruction.id == "clean":
        return Arm.SCENE
    return Arm.JOINT


def run_factorial(
    adapter: EnvironmentAdapter,
    tasks: Iterable[TaskSpec],
    seeds: Iterable[int],
    instructions: Iterable[InstructionVariant],
    scenes: Iterable[ScenePatch],
    semantic_distance_max: float,
) -> list[EpisodeResult]:
    rows: list[EpisodeResult] = []
    for task, seed, instruction, scene in product(tasks, seeds, instructions, scenes):
        feasible = scene_is_feasible(scene)
        semantic_preserved = instruction_preserves_task(instruction, semantic_distance_max)
        if not feasible or not semantic_preserved:
            continue
        signals = adapter.rollout(task, seed, instruction, scene)
        rows.append(
            EpisodeResult(
                task=task.id,
                seed=seed,
                arm=classify_arm(instruction, scene),
                instruction_id=instruction.id,
                instruction=instruction.text,
                scene_id=scene.id,
                scene_kind=scene.kind,
                task_success=signals.task_success,
                feasible=feasible,
                semantic_preserved=semantic_preserved,
                unsafe=is_unsafe(signals, task),
                safety_cost=signals.safety_cost,
                event_step=signals.event_step,
                safety_predicate=task.safety_predicate,
                provenance=signals.extra,
            )
        )
    return rows
