from __future__ import annotations

import hashlib

from vla_xrt.adapters.base import EnvironmentAdapter, RolloutSignals
from vla_xrt.core.types import InstructionVariant, ScenePatch, TaskSpec


class DeterministicMockAdapter(EnvironmentAdapter):
    """A protocol fixture, deliberately not a claim about any physical policy."""

    name = "deterministic-mock-vla"

    def rollout(
        self,
        task: TaskSpec,
        seed: int,
        instruction: InstructionVariant,
        scene: ScenePatch,
    ) -> RolloutSignals:
        key = f"{task.id}:{seed}:{instruction.id}:{scene.id}".encode()
        jitter = int(hashlib.sha256(key).hexdigest()[:4], 16) / 0xFFFF
        joint_trigger = instruction.id == "precision" and scene.id == "route_bystander"
        weak_scene_risk = scene.id == "route_bystander" and instruction.id == "paraphrase"
        safety_cost = 1.25 + 0.1 * jitter if joint_trigger else 0.18 + 0.1 * jitter
        if instruction.id == "precision" and scene.id == "clean":
            safety_cost += 0.08
        if weak_scene_risk:
            safety_cost += 0.15
        unsafe = safety_cost >= task.safety_threshold
        return RolloutSignals(
            task_success=not unsafe,
            safety_cost=round(safety_cost, 4),
            event_step=74 if unsafe else None,
            extra={"adapter": self.name, "fixture": True},
        )
