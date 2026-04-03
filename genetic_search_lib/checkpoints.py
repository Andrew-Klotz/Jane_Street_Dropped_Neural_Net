from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from genetic_search_lib.config import SearchState
from genetic_search_lib.permutations import decode_permutation, encode_permutation


def save_checkpoint(
    state: SearchState,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "latest.pt"
    summary_path = artifact_dir / "summary.json"
    torch.save(
        {
            "population": [encode_permutation(permutation) for permutation in state.population],
            "population_losses": list(state.population_losses),
            "loss_cache": state.loss_cache,
            "best_permutation": encode_permutation(state.best_permutation),
            "best_loss": state.best_loss,
            "generation": state.generation,
            "block_size": state.block_size,
            "saved_at": time.time(),
        },
        checkpoint_path,
    )

    top_entries = sorted(zip(state.population, state.population_losses), key=lambda item: item[1])[:10]
    summary = {
        "generation": state.generation,
        "block_size": state.block_size,
        "best_loss": state.best_loss,
        "top_population": [
            {
                "loss": loss,
                "permutation_prefix": list(permutation[:10]),
                "permutation_suffix": list(permutation[-5:]),
            }
            for permutation, loss in top_entries
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_checkpoint(artifact_dir: Path) -> SearchState | None:
    checkpoint_path = artifact_dir / "latest.pt"
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return SearchState(
        population=[
            decode_permutation(permutation_payload)
            for permutation_payload in payload["population"]
        ],
        population_losses=[float(loss) for loss in payload["population_losses"]],
        loss_cache=payload["loss_cache"],
        best_permutation=decode_permutation(payload["best_permutation"]),
        best_loss=float(payload["best_loss"]),
        generation=int(payload["generation"]),
        block_size=int(payload["block_size"]),
    )
