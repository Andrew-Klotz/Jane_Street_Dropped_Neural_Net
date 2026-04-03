from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PIECES_DIR = DATA_DIR / "pieces"
ARTIFACTS_DIR = ROOT / "artifacts" / "genetic_search"

INPUT_DIM = 48
HIDDEN_DIM = 96
ALT_LENGTH = 96
FINAL_INDEX = 96
INITIAL_SAMPLE_SIZE = 256
REFINE_SAMPLE_SIZE = 2048
TOP_K_INITIAL = 32
TOP_K_FULL_4 = 4
TOP_K_FULL_8 = 8
SMALL_DELTA_RATIO = 1e-2
HUBER_DELTA = 1.0
DEFAULT_POPULATION_SIZE = 64
DEFAULT_SURVIVOR_COUNT = 32
DEFAULT_RANDOM_INJECTIONS = 16
DEFAULT_COMBINATION_CHILDREN = 16
DEFAULT_MAX_BLOCK_SIZE = 8
CHECKPOINT_EVERY = 1
RELATIVE_WEIGHTING = 0.5
STAGNATION_TRIGGER_GENERATIONS = 5
STAGNATION_PROBE_BLOCK_SIZES = (1, 2, 4, 8, 12, 16, 24, 32)
STAGNATION_PROBE_ELITES = 8
BUBBLE_SORT_ELITES = 4


@dataclass(frozen=True)
class AffineLayer:
    layer_id: str
    weight: torch.Tensor
    bias: torch.Tensor
    in_dim: int
    out_dim: int


@dataclass
class DatasetBundle:
    inputs: torch.Tensor
    pred: torch.Tensor
    sample_full: torch.Tensor


@dataclass
class SearchState:
    population: list[tuple[str, ...]]
    population_losses: list[float]
    loss_cache: dict[bytes, float]
    best_permutation: tuple[str, ...]
    best_loss: float
    generation: int
    block_size: int


@dataclass(frozen=True)
class LayerStore:
    layers: dict[str, AffineLayer]
    even_layers: tuple[str, ...]
    odd_layers: tuple[str, ...]
    final_layer_id: str
    even_index: dict[str, int]
    odd_index: dict[str, int]
    even_weights: torch.Tensor
    even_biases: torch.Tensor
    odd_weights: torch.Tensor
    odd_biases: torch.Tensor
    final_weight: torch.Tensor
    final_bias: torch.Tensor
