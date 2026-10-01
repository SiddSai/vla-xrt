from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vla_xrt.adapters.base import EnvironmentAdapter
from vla_xrt.core.types import InstructionVariant, ScenePatch, TaskSpec
from vla_xrt.oracles.gates import instruction_preserves_task, scene_is_feasible


@dataclass(frozen=True)
class SearchObservation:
    instruction_id: str
    scene_id: str
    safety_cost: float


@dataclass(frozen=True)
class SearchResult:
    instruction: InstructionVariant
    scene: ScenePatch
    observations: tuple[SearchObservation, ...]


def alternating_search(
    adapter: EnvironmentAdapter,
    task: TaskSpec,
    seed: int,
    instructions: Iterable[InstructionVariant],
    scenes: Iterable[ScenePatch],
    semantic_distance_max: float,
    rounds: int = 2,
) -> SearchResult:
    """Budgeted coordinate search over a finite, feasible candidate bank.

    The initial implementation is deliberately transparent: it alternates the
    instruction and scene coordinates while retaining every policy query. A
    future mutator can expand either bank without changing this contract.
    """
    allowed_instructions = [
        item for item in instructions if instruction_preserves_task(item, semantic_distance_max)
    ]
    allowed_scenes = [item for item in scenes if scene_is_feasible(item)]
    if not allowed_instructions or not allowed_scenes:
        raise ValueError("search needs at least one feasible instruction and scene")
    clean_instruction = next((item for item in allowed_instructions if item.id == "clean"), allowed_instructions[0])
    clean_scene = next((item for item in allowed_scenes if item.id == "clean"), allowed_scenes[0])
    chosen_instruction, chosen_scene = clean_instruction, clean_scene
    observations: list[SearchObservation] = []

    def best_instruction(scene: ScenePatch) -> InstructionVariant:
        scored: list[tuple[float, InstructionVariant]] = []
        for instruction in allowed_instructions:
            score = adapter.rollout(task, seed, instruction, scene).safety_cost
            observations.append(SearchObservation(instruction.id, scene.id, score))
            scored.append((score, instruction))
        return max(scored, key=lambda item: item[0])[1]

    def best_scene(instruction: InstructionVariant) -> ScenePatch:
        scored: list[tuple[float, ScenePatch]] = []
        for scene in allowed_scenes:
            score = adapter.rollout(task, seed, instruction, scene).safety_cost
            observations.append(SearchObservation(instruction.id, scene.id, score))
            scored.append((score, scene))
        return max(scored, key=lambda item: item[0])[1]

    for _ in range(rounds):
        chosen_instruction = best_instruction(chosen_scene)
        chosen_scene = best_scene(chosen_instruction)
    return SearchResult(chosen_instruction, chosen_scene, tuple(observations))
