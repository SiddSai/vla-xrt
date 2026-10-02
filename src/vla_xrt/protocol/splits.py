from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSplit:
    search_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]

    def validate(self) -> None:
        overlap = set(self.search_seeds) & set(self.heldout_seeds)
        if overlap:
            raise ValueError(f"search and held-out seeds overlap: {sorted(overlap)}")
        if not self.search_seeds or not self.heldout_seeds:
            raise ValueError("both search and held-out splits must be non-empty")
