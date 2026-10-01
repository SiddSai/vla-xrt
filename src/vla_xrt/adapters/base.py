from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vla_xrt.core.types import InstructionVariant, ScenePatch, TaskSpec


@dataclass(frozen=True)
class RolloutSignals:
    task_success: bool
    safety_cost: float
    event_step: int | None
    extra: dict[str, Any]


class EnvironmentAdapter(ABC):
    """A policy/simulator boundary. Implementations own all model preprocessing."""

    name: str

    @abstractmethod
    def rollout(
        self,
        task: TaskSpec,
        seed: int,
        instruction: InstructionVariant,
        scene: ScenePatch,
    ) -> RolloutSignals:
        """Run a closed-loop episode and return simulator-derived signals."""
