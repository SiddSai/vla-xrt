"""A narrow real-policy adapter for pinned OpenVLA + LIBERO deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from vla_xrt.adapters.base import EnvironmentAdapter, RolloutSignals
from vla_xrt.core.types import InstructionVariant, ScenePatch, TaskSpec
from vla_xrt.oracles.non_target_motion import NonTargetMotionOracle


class LiberoEnvironment(Protocol):
    def reset(self, seed: int) -> Mapping[str, object]: ...
    def step(self, action: object) -> tuple[Mapping[str, object], float, bool, Mapping[str, object]]: ...
    def apply_scene_patch(self, patch: ScenePatch) -> None: ...
    def object_positions(self) -> Mapping[str, Sequence[float]]: ...
    def close(self) -> None: ...


PolicyInference = Callable[[Mapping[str, object], str], object]
EnvironmentFactory = Callable[[str], LiberoEnvironment]


@dataclass
class LiberoOpenVLAAdapter(EnvironmentAdapter):
    """Closed-loop adapter with a fixed external OpenVLA policy wrapper.

    ``policy`` retains its upstream OpenVLA prompt template. VLA-XRT passes
    only the task instruction; it never prepends an attacker/system prompt.
    """

    environment_factory: EnvironmentFactory
    policy: PolicyInference
    motion_oracle: NonTargetMotionOracle
    max_steps: int = 280
    name: str = "libero-openvla"

    def rollout(
        self,
        task: TaskSpec,
        seed: int,
        instruction: InstructionVariant,
        scene: ScenePatch,
    ) -> RolloutSignals:
        env = self.environment_factory(task.id)
        try:
            observation = env.reset(seed)
            env.apply_scene_patch(scene)
            initial_positions = env.object_positions()
            max_displacement = 0.0
            event_step: int | None = None
            task_success = False
            for step in range(self.max_steps):
                action = self.policy(observation, instruction.text)
                observation, _, terminated, info = env.step(action)
                displacement = self.motion_oracle.displacement_cm(initial_positions, env.object_positions())
                max_displacement = max(max_displacement, displacement)
                if event_step is None and displacement >= self.motion_oracle.threshold_cm:
                    event_step = step
                task_success = bool(info.get("task_success", False))
                if terminated or task_success:
                    break
            return RolloutSignals(
                task_success=task_success,
                safety_cost=max_displacement,
                event_step=event_step,
                extra={"adapter": self.name, "motion_object": self.motion_oracle.object_name,
                       "motion_threshold_cm": self.motion_oracle.threshold_cm, "max_steps": self.max_steps},
            )
        finally:
            env.close()
