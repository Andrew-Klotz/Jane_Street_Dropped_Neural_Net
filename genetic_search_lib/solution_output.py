from __future__ import annotations

from pathlib import Path
from typing import Sequence


def layer_id_to_piece_index(layer_id: str) -> int:
    if layer_id.startswith("piece_"):
        return int(layer_id.removeprefix("piece_"))
    return int(layer_id)


def permutation_to_submission_indices(permutation: Sequence[str]) -> list[int]:
    return [layer_id_to_piece_index(layer_id) for layer_id in permutation]


def permutation_to_submission_string(permutation: Sequence[str]) -> str:
    return ",".join(str(piece_idx) for piece_idx in permutation_to_submission_indices(permutation))


def save_solution_file(permutation: Sequence[str], artifact_dir: Path, filename: str = "solution.txt") -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    solution_path = artifact_dir / filename
    solution_path.write_text(permutation_to_submission_string(permutation), encoding="utf-8")
    return solution_path
