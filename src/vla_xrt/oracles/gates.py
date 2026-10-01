from vla_xrt.adapters.base import RolloutSignals
from vla_xrt.core.types import InstructionVariant, ScenePatch, TaskSpec


def scene_is_feasible(scene: ScenePatch) -> bool:
    return scene.feasible


def instruction_preserves_task(instruction: InstructionVariant, max_distance: float) -> bool:
    return instruction.semantic_distance <= max_distance


def is_unsafe(signals: RolloutSignals, task: TaskSpec) -> bool:
    return signals.safety_cost >= task.safety_threshold
