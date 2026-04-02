from __future__ import annotations

import csv
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PIECES_DIR = ROOT / "data" / "pieces"
OUTPUT_PATH = ROOT / "data" / "layer_sizes.csv"


def inspect_piece(piece_path: Path) -> dict[str, object]:
    state_dict = torch.load(piece_path, map_location="cpu", weights_only=True)

    weight = state_dict["weight"]
    bias = state_dict.get("bias")

    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D linear weight in {piece_path.name}, got {tuple(weight.shape)}")

    output_size, input_size = weight.shape
    bias_shape = tuple(bias.shape) if bias is not None else ()

    return {
        "piece": piece_path.name,
        "input_size": int(input_size),
        "output_size": int(output_size),
        "weight_shape": "x".join(str(dim) for dim in weight.shape),
        "bias_shape": "x".join(str(dim) for dim in bias_shape),
    }


def main() -> None:
    rows = [inspect_piece(piece_path) for piece_path in sorted(PIECES_DIR.glob("*.pth"))]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["piece", "input_size", "output_size", "weight_shape", "bias_shape"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} layer records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
