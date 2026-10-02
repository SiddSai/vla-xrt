"""Small, inspectable reset-time scene mutations for LIBERO / robosuite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MuJoCoLike(Protocol):
    data: object

    def forward(self) -> None: ...


@dataclass(frozen=True)
class FreeJointPosePatch:
    """Move one existing non-target object before the first observation."""

    id: str
    joint_name: str
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    region_name: str = "approved_tabletop"

    def qpos(self) -> tuple[float, ...]:
        return self.position_xyz + self.quaternion_wxyz


def apply_free_joint_pose(sim: MuJoCoLike, patch: FreeJointPosePatch) -> None:
    """Apply a MuJoCo free-joint pose, then make it visible to rendering/physics."""
    setter = getattr(sim.data, "set_joint_qpos", None)
    if setter is None:
        raise TypeError("sim.data must expose set_joint_qpos for free-joint patches")
    setter(patch.joint_name, patch.qpos())
    sim.forward()
