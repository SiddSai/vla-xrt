"""Small, auditable MuJoCo free-joint scene patches for LIBERO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class MuJoCoSim(Protocol):
    class data:  # noqa: N801 - mirrors MuJoCo's public API
        @staticmethod
        def set_joint_qpos(name: str, value: Sequence[float]) -> None: ...

    def forward(self) -> None: ...


@dataclass(frozen=True)
class FreeJointPosePatch:
    """A seven-value free-joint pose: xyz followed by quaternion wxyz."""

    joint_name: str
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "FreeJointPosePatch":
        position = tuple(float(item) for item in value["position_xyz"])  # type: ignore[index]
        quaternion = tuple(float(item) for item in value["quaternion_wxyz"])  # type: ignore[index]
        if len(position) != 3 or len(quaternion) != 4:
            raise ValueError("A free-joint patch requires 3 xyz values and 4 quaternion values")
        return cls(str(value["joint_name"]), position, quaternion)  # type: ignore[arg-type,index]


def apply_free_joint_pose(sim: MuJoCoSim, patch: FreeJointPosePatch) -> None:
    """Set a free-joint pose and recompute derived MuJoCo state."""

    sim.data.set_joint_qpos(patch.joint_name, (*patch.position_xyz, *patch.quaternion_wxyz))
    sim.forward()
