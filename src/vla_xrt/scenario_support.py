"""Dependency-free geometry helpers shared by scenario patches and oracles."""

from math import sqrt
from typing import Sequence


def euclidean_cm(before_xyz: Sequence[float], after_xyz: Sequence[float]) -> float:
    if len(before_xyz) != 3 or len(after_xyz) != 3:
        raise ValueError("object positions must contain exactly three coordinates")
    return 100.0 * sqrt(sum((a - b) ** 2 for a, b in zip(before_xyz, after_xyz, strict=True)))
