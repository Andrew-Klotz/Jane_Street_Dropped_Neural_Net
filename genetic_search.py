from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parent
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genetic/local search for Jane Street dropped neural net.")
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--survivors", type=int, default=DEFAULT_SURVIVOR_COUNT)
    parser.add_argument("--random-injections", type=int, default=DEFAULT_RANDOM_INJECTIONS)
    parser.add_argument("--combination-children", type=int, default=DEFAULT_COMBINATION_CHILDREN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-block-size", type=int, default=DEFAULT_MAX_BLOCK_SIZE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    parser.add_argument("--top-k-initial", type=int, default=TOP_K_INITIAL)
    parser.add_argument("--top-k-full-4", type=int, default=TOP_K_FULL_4)
    parser.add_argument("--top-k-full-8", type=int, default=TOP_K_FULL_8)
    parser.add_argument("--enable-elite-bubble-search", action="store_true")
    return parser.parse_args()


def get_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def huber_loss_from_predictions(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    errors = predictions - target[None, :]
    abs_errors = errors.abs()
    quadratic = abs_errors.clamp(max=HUBER_DELTA)
    linear = abs_errors - quadratic
    losses = 0.5 * quadratic.square() + HUBER_DELTA * linear
    return losses.mean(dim=1)


def scaled_small_delta(loss_scale: float, ratio: float = SMALL_DELTA_RATIO) -> float:
    return max(1e-4, abs(loss_scale) * ratio)


def load_dataset(device: torch.device) -> DatasetBundle:
    rows: list[list[float]] = []
    with (DATA_DIR / "historical_data.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row[f"measurement_{idx}"]) for idx in range(INPUT_DIM)] + [float(row["pred"])])

    data = torch.tensor(rows, dtype=torch.float32, device=device)
    inputs = data[:, :INPUT_DIM]
    pred = data[:, INPUT_DIM]

    return DatasetBundle(
        inputs=inputs,
        pred=pred,
        sample_full=torch.arange(inputs.shape[0], device=device),
    )


def random_sample_indices(dataset: DatasetBundle, sample_size: int, generator: torch.Generator) -> torch.Tensor:
    sample_size = min(sample_size, dataset.inputs.shape[0])
    indices = torch.randperm(dataset.inputs.shape[0], generator=generator)[:sample_size]
    return indices.to(device=dataset.inputs.device)


def load_layers(device: torch.device) -> LayerStore:
    even_layers: list[str] = []
    odd_layers: list[str] = []
    final_layer_id: str | None = None
    layers: dict[str, AffineLayer] = {}

    for piece_path in sorted(PIECES_DIR.glob("*.pth")):
        state_dict = torch.load(piece_path, map_location=device, weights_only=True)
        weight = state_dict["weight"].to(device=device, dtype=torch.float32)
        bias = state_dict["bias"].to(device=device, dtype=torch.float32)
        out_dim, in_dim = weight.shape
        layer = AffineLayer(
            layer_id=piece_path.stem,
            weight=weight,
            bias=bias,
            in_dim=int(in_dim),
            out_dim=int(out_dim),
        )
        layers[layer.layer_id] = layer

        if (in_dim, out_dim) == (INPUT_DIM, HIDDEN_DIM):
            even_layers.append(layer.layer_id)
        elif (in_dim, out_dim) == (HIDDEN_DIM, INPUT_DIM):
            odd_layers.append(layer.layer_id)
        elif (in_dim, out_dim) == (INPUT_DIM, 1):
            final_layer_id = layer.layer_id
        else:
            raise ValueError(f"Unexpected layer shape {in_dim}->{out_dim} for {piece_path.name}")

    if len(even_layers) != 48 or len(odd_layers) != 48 or final_layer_id is None:
        raise ValueError("Unexpected piece counts; expected 48 even, 48 odd, and 1 final layer.")

    even_weights = torch.stack([layers[layer_id].weight for layer_id in even_layers], dim=0)
    even_biases = torch.stack([layers[layer_id].bias for layer_id in even_layers], dim=0)
    odd_weights = torch.stack([layers[layer_id].weight for layer_id in odd_layers], dim=0)
    odd_biases = torch.stack([layers[layer_id].bias for layer_id in odd_layers], dim=0)
    final_weight = layers[final_layer_id].weight
    final_bias = layers[final_layer_id].bias

    return LayerStore(
        layers=layers,
        even_layers=tuple(even_layers),
        odd_layers=tuple(odd_layers),
        final_layer_id=final_layer_id,
        even_index={layer_id: idx for idx, layer_id in enumerate(even_layers)},
        odd_index={layer_id: idx for idx, layer_id in enumerate(odd_layers)},
        even_weights=even_weights,
        even_biases=even_biases,
        odd_weights=odd_weights,
        odd_biases=odd_biases,
        final_weight=final_weight,
        final_bias=final_bias,
    )


def is_valid_permutation(permutation: Sequence[str], layer_store: LayerStore) -> bool:
    if len(permutation) != FINAL_INDEX + 1:
        return False
    if permutation[-1] != layer_store.final_layer_id:
        return False
    current_dim = INPUT_DIM
    for layer_id in permutation:
        layer = layer_store.layers[layer_id]
        if layer.in_dim != current_dim:
            return False
        current_dim = layer.out_dim
    return current_dim == 1


def split_by_parity(permutation: Sequence[str], final_layer_id: str) -> tuple[list[str], list[str]]:
    body = permutation[:-1]
    if permutation[-1] != final_layer_id:
        raise ValueError("Final layer is not fixed.")
    return list(body[0::2]), list(body[1::2])


def build_permutation(even_sequence: Sequence[str], odd_sequence: Sequence[str], final_layer_id: str) -> tuple[str, ...]:
    if len(even_sequence) != len(odd_sequence):
        raise ValueError(
            f"Permutation requires balanced even/odd layers, got {len(even_sequence)} even and {len(odd_sequence)} odd"
        )
    body: list[str] = []
    for idx in range(len(even_sequence)):
        body.append(even_sequence[idx])
        body.append(odd_sequence[idx])
    return tuple(body + [final_layer_id])


def zero_improvement_meta(baseline_full_loss: float) -> dict[str, float | int | None]:
    return {
        "baseline_loss": baseline_full_loss,
        "best_loss": baseline_full_loss,
        "improvement": 0.0,
        "start_a": None,
        "start_b": None,
        "finalists": 0,
    }


def encode_permutation(permutation: Sequence[str]) -> bytes:
    return b"\0".join(layer_id.encode("utf-8") for layer_id in permutation)


def decode_permutation(payload: bytes) -> tuple[str, ...]:
    return tuple(part.decode("utf-8") for part in payload.split(b"\0"))


def index_sequences_to_permutation(
    even_indices: Sequence[int],
    odd_indices: Sequence[int],
    layer_store: LayerStore,
) -> tuple[str, ...]:
    even_layers = [layer_store.even_layers[idx] for idx in even_indices]
    odd_layers = [layer_store.odd_layers[idx] for idx in odd_indices]
    return build_permutation(even_layers, odd_layers, layer_store.final_layer_id)


def initialize_search(
    population_size: int,
    even_layers: Sequence[str],
    odd_layers: Sequence[str],
    final_layer_id: str,
    rng: random.Random,
    excluded: set[bytes] | None = None,
) -> list[tuple[str, ...]]:
    seen: set[bytes] = set()
    excluded = excluded or set()
    population: list[tuple[str, ...]] = []

    while len(population) < population_size:
        even_perm = list(even_layers)
        odd_perm = list(odd_layers)
        rng.shuffle(even_perm)
        rng.shuffle(odd_perm)
        permutation = build_permutation(even_perm, odd_perm, final_layer_id)
        permutation_key = encode_permutation(permutation)
        if permutation_key not in seen and permutation_key not in excluded:
            seen.add(permutation_key)
            population.append(permutation)

    return population


def permutation_to_index_sequences(
    permutation: Sequence[str],
    layer_store: LayerStore,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    even_indices = torch.tensor(
        [layer_store.even_index[layer_id] for layer_id in permutation[:-1:2]],
        dtype=torch.long,
        device=device,
    )
    odd_indices = torch.tensor(
        [layer_store.odd_index[layer_id] for layer_id in permutation[1:-1:2]],
        dtype=torch.long,
        device=device,
    )
    return even_indices, odd_indices


def permutations_to_index_sequences(
    permutations: Sequence[Sequence[str]],
    layer_store: LayerStore,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    even_indices = torch.tensor(
        [[layer_store.even_index[layer_id] for layer_id in permutation[:-1:2]] for permutation in permutations],
        dtype=torch.long,
        device=device,
    )
    odd_indices = torch.tensor(
        [[layer_store.odd_index[layer_id] for layer_id in permutation[1:-1:2]] for permutation in permutations],
        dtype=torch.long,
        device=device,
    )
    return even_indices, odd_indices


def forward_population_indices(
    even_indices: torch.Tensor,
    odd_indices: torch.Tensor,
    layer_store: LayerStore,
    inputs: torch.Tensor,
) -> torch.Tensor:
    state = inputs.unsqueeze(0).expand(even_indices.shape[0], -1, -1)

    for block_idx in range(even_indices.shape[1]):
        inp_weights = layer_store.even_weights.index_select(0, even_indices[:, block_idx])
        inp_biases = layer_store.even_biases.index_select(0, even_indices[:, block_idx])
        hidden = torch.bmm(state, inp_weights.transpose(1, 2)) + inp_biases.unsqueeze(1)
        hidden = torch.relu(hidden)

        out_weights = layer_store.odd_weights.index_select(0, odd_indices[:, block_idx])
        out_biases = layer_store.odd_biases.index_select(0, odd_indices[:, block_idx])
        delta = torch.bmm(hidden, out_weights.transpose(1, 2)) + out_biases.unsqueeze(1)
        state = state + delta

    predictions = torch.matmul(state, layer_store.final_weight.t()) + layer_store.final_bias
    return predictions.squeeze(-1)


def forward_from_boundary(
    even_indices: torch.Tensor,
    odd_indices: torch.Tensor,
    layer_store: LayerStore,
    start_states: torch.Tensor,
    start_layer: int,
    end_boundary_layer: int,
) -> torch.Tensor:
    state = start_states
    start_block = start_layer // 2
    end_block = end_boundary_layer // 2
    if start_block >= end_block:
        return state

    gathered_even_weights = layer_store.even_weights[even_indices[:, start_block:end_block]]
    gathered_even_biases = layer_store.even_biases[even_indices[:, start_block:end_block]]
    gathered_odd_weights = layer_store.odd_weights[odd_indices[:, start_block:end_block]]
    gathered_odd_biases = layer_store.odd_biases[odd_indices[:, start_block:end_block]]

    for local_block in range(end_block - start_block):
        inp_weights = gathered_even_weights[:, local_block]
        inp_biases = gathered_even_biases[:, local_block]
        hidden = torch.bmm(state, inp_weights.transpose(1, 2)) + inp_biases.unsqueeze(1)
        hidden = torch.relu(hidden)

        out_weights = gathered_odd_weights[:, local_block]
        out_biases = gathered_odd_biases[:, local_block]
        delta = torch.bmm(hidden, out_weights.transpose(1, 2)) + out_biases.unsqueeze(1)
        state = state + delta

    return state


def forward_shared_tail_from_boundary(
    even_indices: torch.Tensor,
    odd_indices: torch.Tensor,
    layer_store: LayerStore,
    boundary_states: torch.Tensor,
    start_boundary_layer: int,
) -> torch.Tensor:
    state = boundary_states
    start_block = start_boundary_layer // 2
    gathered_even_weights = layer_store.even_weights[even_indices[start_block:]]
    gathered_even_biases = layer_store.even_biases[even_indices[start_block:]]
    gathered_odd_weights = layer_store.odd_weights[odd_indices[start_block:]]
    gathered_odd_biases = layer_store.odd_biases[odd_indices[start_block:]]

    for local_block in range(gathered_even_weights.shape[0]):
        inp_weights = gathered_even_weights[local_block]
        inp_biases = gathered_even_biases[local_block]
        hidden = torch.matmul(state, inp_weights.t()) + inp_biases
        hidden = torch.relu(hidden)

        out_weights = gathered_odd_weights[local_block]
        out_biases = gathered_odd_biases[local_block]
        delta = torch.matmul(hidden, out_weights.t()) + out_biases
        state = state + delta

    predictions = torch.matmul(state, layer_store.final_weight.t()) + layer_store.final_bias
    return predictions.squeeze(-1)


def prefix_states_for_permutation(
    even_indices: torch.Tensor,
    odd_indices: torch.Tensor,
    layer_store: LayerStore,
    sample_inputs: torch.Tensor,
) -> list[torch.Tensor]:
    states: list[torch.Tensor] = [sample_inputs]
    state = sample_inputs
    for block_idx in range(ALT_LENGTH // 2):
        inp_weights = layer_store.even_weights[even_indices[block_idx]]
        inp_biases = layer_store.even_biases[even_indices[block_idx]]
        hidden = torch.relu(torch.matmul(state, inp_weights.t()) + inp_biases)
        out_weights = layer_store.odd_weights[odd_indices[block_idx]]
        out_biases = layer_store.odd_biases[odd_indices[block_idx]]
        state = state + torch.matmul(hidden, out_weights.t()) + out_biases
        states.append(state)
    return states


@torch.inference_mode()
def evaluate_permutations(
    permutations: Sequence[Sequence[str]],
    layer_store: LayerStore,
    dataset: DatasetBundle,
    sample_indices: torch.Tensor,
    batch_size: int = 64,
) -> list[float]:
    inputs = dataset.inputs.index_select(0, sample_indices)
    target_tensor = dataset.pred.index_select(0, sample_indices)
    losses: list[float] = []

    for start in range(0, len(permutations), batch_size):
        permutation_batch = permutations[start : start + batch_size]
        even_indices, odd_indices = permutations_to_index_sequences(permutation_batch, layer_store, inputs.device)
        predictions = forward_population_indices(even_indices, odd_indices, layer_store, inputs)
        batch_losses = huber_loss_from_predictions(predictions, target_tensor)
        losses.extend(batch_losses.detach().cpu().tolist())

    return losses


@torch.inference_mode()
def best_swap(
    permutation: Sequence[str],
    layer_store: LayerStore,
    dataset: DatasetBundle,
    block_size: int,
    baseline_full_loss: float,
    sample_generator: torch.Generator,
    top_k_initial: int = TOP_K_INITIAL,
    top_k_full_4: int = TOP_K_FULL_4,
    top_k_full_8: int = TOP_K_FULL_8,
) -> tuple[tuple[str, ...], dict[str, float | int | None]]:
    if block_size < 1 or block_size > ALT_LENGTH:
        raise ValueError(f"Invalid block size {block_size}")

    sample_256 = random_sample_indices(dataset, INITIAL_SAMPLE_SIZE, sample_generator)
    sample_2048 = random_sample_indices(dataset, REFINE_SAMPLE_SIZE, sample_generator)
    candidate_meta: list[tuple[int, int, int]] = []
    perm_even_indices, perm_odd_indices = permutation_to_index_sequences(permutation, layer_store, dataset.inputs.device)
    base_even = perm_even_indices.detach().cpu().tolist()
    base_odd = perm_odd_indices.detach().cpu().tolist()
    candidate_even_rows: list[list[int]] = []
    candidate_odd_rows: list[list[int]] = []
    full_start_specs: list[tuple[int, int, tuple[tuple[int, int, int], ...]]] = []

    last_start = ALT_LENGTH - block_size
    for start_a in range(0, last_start + 1):
        group_row_start = len(candidate_meta)
        finish_specs: list[tuple[int, int, int]] = []
        for start_b in range(start_a + block_size, last_start + 1):
            if (start_a % 2) != (start_b % 2):
                continue
            candidate_even = list(base_even)
            candidate_odd = list(base_odd)

            if start_a % 2 == 0:
                even_start_a = start_a // 2
                even_start_b = start_b // 2
                even_count = (block_size + 1) // 2
                odd_start_a = start_a // 2
                odd_start_b = start_b // 2
                odd_count = block_size // 2
            else:
                odd_start_a = start_a // 2
                odd_start_b = start_b // 2
                odd_count = (block_size + 1) // 2
                even_start_a = (start_a + 1) // 2
                even_start_b = (start_b + 1) // 2
                even_count = block_size // 2

            if even_count > 0:
                even_block_a = candidate_even[even_start_a : even_start_a + even_count]
                even_block_b = candidate_even[even_start_b : even_start_b + even_count]
                candidate_even[even_start_a : even_start_a + even_count] = even_block_b
                candidate_even[even_start_b : even_start_b + even_count] = even_block_a

            if odd_count > 0:
                odd_block_a = candidate_odd[odd_start_a : odd_start_a + odd_count]
                odd_block_b = candidate_odd[odd_start_b : odd_start_b + odd_count]
                candidate_odd[odd_start_a : odd_start_a + odd_count] = odd_block_b
                candidate_odd[odd_start_b : odd_start_b + odd_count] = odd_block_a

            tail_start = start_b + block_size
            end_boundary = tail_start if tail_start % 2 == 0 else tail_start + 1
            candidate_meta.append((start_a, start_b, end_boundary))
            candidate_even_rows.append(candidate_even)
            candidate_odd_rows.append(candidate_odd)
            group_row_end = len(candidate_meta)
            if finish_specs and finish_specs[-1][0] == end_boundary and finish_specs[-1][2] == group_row_end - 1:
                finish_end_boundary, finish_start, _ = finish_specs[-1]
                finish_specs[-1] = (finish_end_boundary, finish_start, group_row_end)
            else:
                finish_specs.append((end_boundary, group_row_end - group_row_start - 1, group_row_end - group_row_start))

        if group_row_start < len(candidate_meta):
            full_start_specs.append((start_a, group_row_start, tuple(finish_specs)))

    if not candidate_meta:
        return tuple(permutation), {
            "baseline_loss": baseline_full_loss,
            "best_loss": baseline_full_loss,
            "improvement": 0.0,
            "start_a": None,
            "start_b": None,
        }

    candidate_even_tensor = torch.tensor(candidate_even_rows, dtype=torch.long, device=dataset.inputs.device)
    candidate_odd_tensor = torch.tensor(candidate_odd_rows, dtype=torch.long, device=dataset.inputs.device)
    unique_end_boundaries = sorted({end_boundary for _, _, end_boundary in candidate_meta})

    def score_all_candidates(indices: torch.Tensor) -> torch.Tensor:
        inputs = dataset.inputs.index_select(0, indices)
        target = dataset.pred.index_select(0, indices)
        prefix_states = prefix_states_for_permutation(perm_even_indices, perm_odd_indices, layer_store, inputs)
        losses = torch.empty(len(candidate_meta), dtype=torch.float32, device=inputs.device)
        boundary_states_by_end: dict[int, list[torch.Tensor]] = {end_boundary: [] for end_boundary in unique_end_boundaries}
        boundary_rows_by_end: dict[int, list[int]] = {end_boundary: [] for end_boundary in unique_end_boundaries}

        for start_a, row_start, finish_specs in full_start_specs:
            row_end = row_start + finish_specs[-1][2]
            prefix_state = prefix_states[start_a // 2]
            start_block = start_a // 2
            max_end_block = finish_specs[-1][0] // 2
            group_even_indices = candidate_even_tensor[row_start:row_end]
            group_odd_indices = candidate_odd_tensor[row_start:row_end]
            state = prefix_state.unsqueeze(0).expand(row_end - row_start, -1, -1)
            gathered_even_weights = layer_store.even_weights[group_even_indices[:, start_block:max_end_block]]
            gathered_even_biases = layer_store.even_biases[group_even_indices[:, start_block:max_end_block]]
            gathered_odd_weights = layer_store.odd_weights[group_odd_indices[:, start_block:max_end_block]]
            gathered_odd_biases = layer_store.odd_biases[group_odd_indices[:, start_block:max_end_block]]

            finish_ptr = 0
            for local_block in range(max_end_block - start_block):
                inp_weights = gathered_even_weights[:, local_block]
                inp_biases = gathered_even_biases[:, local_block]
                hidden = torch.bmm(state, inp_weights.transpose(1, 2)) + inp_biases.unsqueeze(1)
                hidden = torch.relu(hidden)

                out_weights = gathered_odd_weights[:, local_block]
                out_biases = gathered_odd_biases[:, local_block]
                delta = torch.bmm(hidden, out_weights.transpose(1, 2)) + out_biases.unsqueeze(1)
                state = state + delta

                current_boundary = 2 * (start_block + local_block + 1)
                while finish_ptr < len(finish_specs) and finish_specs[finish_ptr][0] == current_boundary:
                    _, rel_start, rel_end = finish_specs[finish_ptr]
                    boundary_states_by_end[current_boundary].append(state[rel_start:rel_end].clone())
                    boundary_rows_by_end[current_boundary].extend(range(row_start + rel_start, row_start + rel_end))
                    finish_ptr += 1

        for end_boundary in unique_end_boundaries:
            if not boundary_states_by_end[end_boundary]:
                continue
            predictions = forward_shared_tail_from_boundary(
                perm_even_indices,
                perm_odd_indices,
                layer_store,
                torch.cat(boundary_states_by_end[end_boundary], dim=0),
                end_boundary,
            )
            end_losses = huber_loss_from_predictions(predictions, target)
            row_tensor = torch.tensor(boundary_rows_by_end[end_boundary], dtype=torch.long, device=inputs.device)
            losses[row_tensor] = end_losses

        return losses

    def score_candidate_subset(indices: torch.Tensor, candidate_indices: Sequence[int]) -> list[float]:
        inputs = dataset.inputs.index_select(0, indices)
        target = dataset.pred.index_select(0, indices)
        prefix_states = prefix_states_for_permutation(perm_even_indices, perm_odd_indices, layer_store, inputs)
        losses_by_index: dict[int, float] = {}
        grouped: dict[int, dict[int, list[int]]] = {}
        for idx in candidate_indices:
            start_a, _, end_boundary = candidate_meta[idx]
            grouped.setdefault(end_boundary, {}).setdefault(start_a, []).append(idx)

        for end_boundary, by_start in grouped.items():
            group_boundary_states: list[torch.Tensor] = []
            group_order: list[int] = []
            for start_a, group_indices in by_start.items():
                start_group_tensor = torch.tensor(group_indices, dtype=torch.long, device=dataset.inputs.device)
                prefix_state = prefix_states[start_a // 2]
                chunk_even_indices = candidate_even_tensor.index_select(0, start_group_tensor)
                chunk_odd_indices = candidate_odd_tensor.index_select(0, start_group_tensor)
                expanded_prefix = prefix_state.unsqueeze(0).expand(start_group_tensor.shape[0], -1, -1)
                middle_states = forward_from_boundary(
                    chunk_even_indices,
                    chunk_odd_indices,
                    layer_store,
                    expanded_prefix,
                    start_a,
                    end_boundary,
                )
                group_boundary_states.append(middle_states)
                group_order.extend(group_indices)

            if group_boundary_states:
                predictions = forward_shared_tail_from_boundary(
                    perm_even_indices,
                    perm_odd_indices,
                    layer_store,
                    torch.cat(group_boundary_states, dim=0),
                    end_boundary,
                )
                losses = huber_loss_from_predictions(predictions, target).detach().cpu().tolist()
                for idx, loss in zip(group_order, losses):
                    losses_by_index[idx] = loss

        return [losses_by_index[idx] for idx in candidate_indices]

    initial_losses = score_all_candidates(sample_256)
    initial_count = min(top_k_initial, initial_losses.shape[0])
    _, top_positions = torch.topk(initial_losses, k=initial_count, largest=False)

    refine_indices = top_positions.detach().cpu().tolist()
    refine_losses = score_candidate_subset(sample_2048, refine_indices)
    ranked_refine = sorted(zip(refine_indices, refine_losses), key=lambda item: item[1])
    relative_small_delta = scaled_small_delta(ranked_refine[0][1]) if ranked_refine else scaled_small_delta(baseline_full_loss)

    if len(ranked_refine) > 1 and (ranked_refine[min(len(ranked_refine), top_k_full_8) - 1][1] - ranked_refine[0][1]) <= relative_small_delta:
        full_count = min(top_k_full_8, len(ranked_refine))
    elif len(ranked_refine) > 1 and (ranked_refine[min(len(ranked_refine), top_k_full_4) - 1][1] - ranked_refine[0][1]) <= relative_small_delta:
        full_count = min(top_k_full_4, len(ranked_refine))
    else:
        full_count = 1

    final_indices = [idx for idx, _ in ranked_refine[:full_count]]
    final_losses = score_candidate_subset(dataset.sample_full, final_indices)
    ranked_final = sorted(zip(final_indices, final_losses), key=lambda item: item[1])
    best_index, best_loss = ranked_final[0]
    start_a, start_b, _ = candidate_meta[best_index]
    best_candidate = index_sequences_to_permutation(
        candidate_even_tensor[best_index].detach().cpu().tolist(),
        candidate_odd_tensor[best_index].detach().cpu().tolist(),
        layer_store,
    )

    if best_loss >= baseline_full_loss:
        return tuple(permutation), {
            "baseline_loss": baseline_full_loss,
            "best_loss": baseline_full_loss,
            "improvement": 0.0,
            "start_a": None,
            "start_b": None,
            "finalists": full_count,
        }

    return best_candidate, {
        "baseline_loss": baseline_full_loss,
        "best_loss": best_loss,
        "improvement": baseline_full_loss - best_loss,
        "start_a": start_a,
        "start_b": start_b,
        "finalists": full_count,
    }


def combination(
    donor: Sequence[str],
    recipient: Sequence[str],
    start: int,
    length: int,
    insert_at: int,
    final_layer_id: str,
) -> tuple[str, ...]:
    if donor[-1] != final_layer_id or recipient[-1] != final_layer_id:
        raise ValueError("Final layer must stay fixed.")
    if start < 0 or length < 1 or start + length > ALT_LENGTH:
        raise ValueError("Invalid donor slice.")
    if insert_at < 0 or insert_at + length > ALT_LENGTH:
        raise ValueError("Invalid insertion point.")
    if (start % 2) != (insert_at % 2):
        raise ValueError("Insertion point must preserve parity.")

    donor_slice = list(donor[start : start + length])
    if start % 2 == 0:
        donor_even = donor_slice[0::2]
        donor_odd = donor_slice[1::2]
    else:
        donor_odd = donor_slice[0::2]
        donor_even = donor_slice[1::2]

    recipient_even, recipient_odd = split_by_parity(recipient, final_layer_id)
    recipient_even = [layer_id for layer_id in recipient_even if layer_id not in donor_even]
    recipient_odd = [layer_id for layer_id in recipient_odd if layer_id not in donor_odd]

    even_insert = (insert_at + 1) // 2
    odd_insert = insert_at // 2

    recipient_even[even_insert:even_insert] = donor_even
    recipient_odd[odd_insert:odd_insert] = donor_odd
    return build_permutation(recipient_even, recipient_odd, final_layer_id)


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


def choose_combination_children(
    survivors: Sequence[tuple[str, ...]],
    count: int,
    final_layer_id: str,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    children: list[tuple[str, ...]] = []
    if len(survivors) < 2 or count <= 0:
        return children

    attempts = 0
    while len(children) < count and attempts < count * 20:
        choices = rng.sample(range(len(survivors)), k=2)
        donor = survivors[choices[0]]
        recipient = survivors[choices[1]]
        start = rng.randrange(0, ALT_LENGTH)
        max_length = min(8, ALT_LENGTH - start)
        length = rng.randint(1, max_length)
        parity_options = [idx for idx in range(0, ALT_LENGTH - length + 1) if (idx % 2) == (start % 2)]
        insert_at = parity_options[rng.randrange(len(parity_options))]
        child = combination(donor, recipient, start, length, insert_at, final_layer_id)
        children.append(child)
        attempts += 1
    return children


def swap_adjacent_residual_blocks(
    permutation: Sequence[str],
    block_idx: int,
    final_layer_id: str,
) -> tuple[str, ...]:
    even_sequence, odd_sequence = split_by_parity(permutation, final_layer_id)
    if block_idx < 0 or block_idx + 1 >= len(even_sequence):
        raise ValueError(f"Invalid residual block index {block_idx}")
    even_sequence[block_idx], even_sequence[block_idx + 1] = even_sequence[block_idx + 1], even_sequence[block_idx]
    odd_sequence[block_idx], odd_sequence[block_idx + 1] = odd_sequence[block_idx + 1], odd_sequence[block_idx]
    return build_permutation(even_sequence, odd_sequence, final_layer_id)


@torch.inference_mode()
def evaluate_adjacent_block_swap_loss(
    perm_even_indices: torch.Tensor,
    perm_odd_indices: torch.Tensor,
    prefix_states: Sequence[torch.Tensor],
    block_idx: int,
    layer_store: LayerStore,
    dataset: DatasetBundle,
) -> float:
    state = prefix_states[block_idx]

    first_even_weight = layer_store.even_weights[perm_even_indices[block_idx + 1]]
    first_even_bias = layer_store.even_biases[perm_even_indices[block_idx + 1]]
    hidden = torch.relu(torch.matmul(state, first_even_weight.t()) + first_even_bias)
    first_odd_weight = layer_store.odd_weights[perm_odd_indices[block_idx + 1]]
    first_odd_bias = layer_store.odd_biases[perm_odd_indices[block_idx + 1]]
    state = state + torch.matmul(hidden, first_odd_weight.t()) + first_odd_bias

    second_even_weight = layer_store.even_weights[perm_even_indices[block_idx]]
    second_even_bias = layer_store.even_biases[perm_even_indices[block_idx]]
    hidden = torch.relu(torch.matmul(state, second_even_weight.t()) + second_even_bias)
    second_odd_weight = layer_store.odd_weights[perm_odd_indices[block_idx]]
    second_odd_bias = layer_store.odd_biases[perm_odd_indices[block_idx]]
    boundary_state = state + torch.matmul(hidden, second_odd_weight.t()) + second_odd_bias

    predictions = forward_shared_tail_from_boundary(
        perm_even_indices,
        perm_odd_indices,
        layer_store,
        boundary_state,
        2 * (block_idx + 2),
    )
    return float(huber_loss_from_predictions(predictions, dataset.pred)[0].item())


def bubble_adjacent_block_search(
    permutation: Sequence[str],
    layer_store: LayerStore,
    dataset: DatasetBundle,
    baseline_full_loss: float,
    loss_cache: dict[bytes, float],
) -> tuple[tuple[str, ...], dict[str, float | int | None]]:
    current_permutation = tuple(permutation)
    current_loss = baseline_full_loss
    adjacent_swaps = 0
    block_count = ALT_LENGTH // 2

    for direction in (range(block_count - 1), range(block_count - 2, -1, -1)):
        perm_even_indices, perm_odd_indices = permutation_to_index_sequences(
            current_permutation,
            layer_store,
            dataset.inputs.device,
        )
        prefix_states = prefix_states_for_permutation(
            perm_even_indices,
            perm_odd_indices,
            layer_store,
            dataset.inputs,
        )

        for block_idx in direction:
            candidate = swap_adjacent_residual_blocks(
                current_permutation,
                block_idx,
                layer_store.final_layer_id,
            )
            candidate_key = encode_permutation(candidate)
            candidate_loss = loss_cache.get(candidate_key)
            if candidate_loss is None:
                candidate_loss = evaluate_adjacent_block_swap_loss(
                    perm_even_indices,
                    perm_odd_indices,
                    prefix_states,
                    block_idx,
                    layer_store,
                    dataset,
                )
                loss_cache[candidate_key] = candidate_loss

            if candidate_loss < current_loss:
                current_permutation = candidate
                current_loss = candidate_loss
                adjacent_swaps += 1
                perm_even_indices, perm_odd_indices = permutation_to_index_sequences(
                    current_permutation,
                    layer_store,
                    dataset.inputs.device,
                )
                prefix_states = prefix_states_for_permutation(
                    perm_even_indices,
                    perm_odd_indices,
                    layer_store,
                    dataset.inputs,
                )

    if current_loss >= baseline_full_loss:
        return tuple(permutation), zero_improvement_meta(baseline_full_loss)

    return current_permutation, {
        "baseline_loss": baseline_full_loss,
        "best_loss": current_loss,
        "improvement": baseline_full_loss - current_loss,
        "start_a": None,
        "start_b": None,
        "finalists": adjacent_swaps,
    }


def run_swap_search(
    permutation: Sequence[str],
    permutation_key: bytes,
    baseline_loss: float,
    layer_store: LayerStore,
    dataset: DatasetBundle,
    max_block_size: int,
    args: argparse.Namespace,
    rng: random.Random,
    torch_generator: torch.Generator,
    is_elite: bool,
    elite_block_exhaustion: dict[bytes, int],
) -> tuple[tuple[str, ...], dict[str, float | int | None], str]:
    exhausted_max_block_size = elite_block_exhaustion.get(permutation_key, 0)
    attempted_block_sizes: list[int] = []

    if is_elite and exhausted_max_block_size >= max_block_size:
        return tuple(permutation), zero_improvement_meta(baseline_loss), f"skip<={exhausted_max_block_size}/{max_block_size}"

    if is_elite and exhausted_max_block_size > 0:
        initial_block_size = max_block_size
        fallback_block_sizes = list(range(max_block_size - 1, exhausted_max_block_size, -1))
    else:
        initial_block_size = rng.randint(1, max_block_size)
        fallback_block_sizes = (
            [
                block_size
                for block_size in range(max_block_size, 0, -1)
                if block_size != initial_block_size
            ]
            if is_elite
            else []
        )

    improved = tuple(permutation)
    meta = zero_improvement_meta(baseline_loss)
    attempted_block_sizes.append(initial_block_size)
    candidate, candidate_meta = best_swap(
        permutation,
        layer_store,
        dataset,
        block_size=initial_block_size,
        baseline_full_loss=baseline_loss,
        sample_generator=torch_generator,
        top_k_initial=args.top_k_initial,
        top_k_full_4=args.top_k_full_4,
        top_k_full_8=args.top_k_full_8,
    )
    meta = candidate_meta
    if float(candidate_meta["improvement"]) > 0.0:
        improved = candidate
        elite_block_exhaustion.pop(permutation_key, None)
        elite_block_exhaustion.pop(encode_permutation(improved), None)
    else:
        for block_size in fallback_block_sizes:
            attempted_block_sizes.append(block_size)
            candidate, candidate_meta = best_swap(
                permutation,
                layer_store,
                dataset,
                block_size=block_size,
                baseline_full_loss=baseline_loss,
                sample_generator=torch_generator,
                top_k_initial=args.top_k_initial,
                top_k_full_4=args.top_k_full_4,
                top_k_full_8=args.top_k_full_8,
            )
            meta = candidate_meta
            if float(candidate_meta["improvement"]) > 0.0:
                improved = candidate
                elite_block_exhaustion.pop(permutation_key, None)
                elite_block_exhaustion.pop(encode_permutation(improved), None)
                break
        else:
            if is_elite:
                elite_block_exhaustion[permutation_key] = max_block_size

    block_display = (
        f"{attempted_block_sizes[0]}/{max_block_size}"
        if len(attempted_block_sizes) == 1
        else f"{','.join(str(size) for size in attempted_block_sizes)}/{max_block_size}"
    )
    return improved, meta, block_display


def improve_population(
    population: Sequence[tuple[str, ...]],
    population_losses: Sequence[float],
    layer_store: LayerStore,
    dataset: DatasetBundle,
    max_block_size: int,
    args: argparse.Namespace,
    rng: random.Random,
    torch_generator: torch.Generator,
    loss_cache: dict[bytes, float],
    elite_keys: set[bytes],
    bubble_elite_keys: set[bytes],
    elite_block_exhaustion: dict[bytes, int],
    bubble_search_exhausted: set[bytes],
) -> tuple[list[tuple[str, ...]], list[float], list[float]]:
    improved_population: list[tuple[str, ...]] = []
    improvements: list[float] = []
    improved_losses: list[float] = []

    for idx, (permutation, baseline_loss) in enumerate(zip(population, population_losses), start=1):
        permutation_key = encode_permutation(permutation)
        is_elite = permutation_key in elite_keys
        search_display = "swap"
        improved = tuple(permutation)
        meta = zero_improvement_meta(baseline_loss)
        block_display = f"n/a/{max_block_size}"
        bubble_attempted = False
        bubble_enabled = args.enable_elite_bubble_search and permutation_key in bubble_elite_keys

        if bubble_enabled and permutation_key not in bubble_search_exhausted:
            bubble_attempted = True
            search_display = "bubble"
            improved, meta = bubble_adjacent_block_search(
                permutation,
                layer_store,
                dataset,
                baseline_loss,
                loss_cache,
            )
            block_display = "adjacent"

        if float(meta["improvement"]) <= 0.0:
            improved, meta, block_display = run_swap_search(
                permutation,
                permutation_key,
                baseline_loss,
                layer_store,
                dataset,
                max_block_size,
                args,
                rng,
                torch_generator,
                is_elite,
                elite_block_exhaustion,
            )
            if bubble_enabled:
                search_display = "bubble-skip->swap" if not bubble_attempted else "bubble->swap"

        if bubble_attempted and float(meta["improvement"]) <= 0.0:
            bubble_search_exhausted.add(permutation_key)

        improved_population.append(improved)
        improvements.append(float(meta["improvement"]))
        improved_losses.append(float(meta["best_loss"]))
        print(
            f"  perm {idx:03d}/{len(population):03d} search={search_display} block={block_display} "
            f"base={meta['baseline_loss']:.6f} best={meta['best_loss']:.6f} "
            f"delta={meta['improvement']:.6f} swap=({meta['start_a']},{meta['start_b']}) finalists={meta.get('finalists')}"
        )

    return improved_population, improvements, improved_losses


def best_swap_among_sizes(
    permutation: Sequence[str],
    layer_store: LayerStore,
    dataset: DatasetBundle,
    block_sizes: Sequence[int],
    baseline_full_loss: float,
    args: argparse.Namespace,
    torch_generator: torch.Generator,
    top_k_initial_override: int | None = None,
) -> tuple[tuple[str, ...], dict[str, float | int | None], int | None]:
    best_permutation = tuple(permutation)
    best_meta: dict[str, float | int | None] = {
        "baseline_loss": baseline_full_loss,
        "best_loss": baseline_full_loss,
        "improvement": 0.0,
        "start_a": None,
        "start_b": None,
        "finalists": 0,
    }
    best_block_size: int | None = None

    for block_size in block_sizes:
        candidate, meta = best_swap(
            permutation,
            layer_store,
            dataset,
            block_size=block_size,
            baseline_full_loss=baseline_full_loss,
            sample_generator=torch_generator,
            top_k_initial=top_k_initial_override if top_k_initial_override is not None else args.top_k_initial,
            top_k_full_4=args.top_k_full_4,
            top_k_full_8=args.top_k_full_8,
        )
        if float(meta["best_loss"]) < float(best_meta["best_loss"]):
            best_permutation = candidate
            best_meta = meta
            best_block_size = block_size

    return best_permutation, best_meta, best_block_size


def apply_stagnation_probe(
    population: Sequence[tuple[str, ...]],
    prior_losses: Sequence[float],
    losses: Sequence[float],
    improvements: Sequence[float],
    layer_store: LayerStore,
    dataset: DatasetBundle,
    args: argparse.Namespace,
    torch_generator: torch.Generator,
) -> tuple[list[tuple[str, ...]], list[float], list[float], list[float]]:
    if not population:
        return list(population), list(prior_losses), list(losses), list(improvements)

    updated_population = list(population)
    updated_losses = list(losses)
    updated_improvements = list(improvements)
    ranked_indices = sorted(range(len(updated_population)), key=lambda idx: updated_losses[idx])
    probe_count = min(STAGNATION_PROBE_ELITES, len(ranked_indices))
    print(
        f"stagnation probe: checking top {probe_count} elites with block sizes "
        f"{','.join(str(size) for size in STAGNATION_PROBE_BLOCK_SIZES)}"
    )

    for elite_rank, population_idx in enumerate(ranked_indices[:probe_count], start=1):
        candidate, meta, block_size = best_swap_among_sizes(
            updated_population[population_idx],
            layer_store,
            dataset,
            STAGNATION_PROBE_BLOCK_SIZES,
            updated_losses[population_idx],
            args,
            torch_generator,
            top_k_initial_override=min(args.top_k_initial * 2, len(updated_population[population_idx])),
        )
        if float(meta["best_loss"]) < updated_losses[population_idx]:
            updated_population[population_idx] = candidate
            updated_losses[population_idx] = float(meta["best_loss"])
            updated_improvements[population_idx] = prior_losses[population_idx] - updated_losses[population_idx]

        block_display = "none" if block_size is None else str(block_size)
        print(
            f"  stagnation elite {elite_rank:02d}/{probe_count:02d} block={block_display} "
            f"base={meta['baseline_loss']:.6f} best={meta['best_loss']:.6f} delta={meta['improvement']:.6f}"
        )

    ranked_indices = sorted(range(len(updated_population)), key=lambda idx: updated_losses[idx])
    elite_count = min(len(updated_population), max(1, min(args.survivors, len(updated_population)) // 2))
    protected_count = min(STAGNATION_PROBE_ELITES, elite_count)
    drop_indices = [
        idx
        for idx in ranked_indices[protected_count:elite_count]
        if updated_improvements[idx] <= 0.0
    ]
    if not drop_indices:
        return updated_population, list(prior_losses), updated_losses, updated_improvements

    drop_set = set(drop_indices)
    print(f"stagnation pruning: dropping {len(drop_indices)} zero-delta elites outside top {protected_count}")
    filtered_population = [permutation for idx, permutation in enumerate(updated_population) if idx not in drop_set]
    filtered_prior_losses = [loss for idx, loss in enumerate(prior_losses) if idx not in drop_set]
    filtered_losses = [loss for idx, loss in enumerate(updated_losses) if idx not in drop_set]
    filtered_improvements = [improvement for idx, improvement in enumerate(updated_improvements) if idx not in drop_set]
    return filtered_population, filtered_prior_losses, filtered_losses, filtered_improvements


def select_survivors(
    population: Sequence[tuple[str, ...]],
    losses: Sequence[float],
    improvements: Sequence[float],
    prior_losses: Sequence[float],
    survivor_count: int,
) -> list[tuple[tuple[str, ...], float]]:
    if survivor_count <= 0 or not population:
        return []

    indexed = list(zip(population, losses, improvements, prior_losses))
    elite_count = min(len(indexed), max(1, survivor_count // 2))
    ranked_by_loss = sorted(indexed, key=lambda item: item[1])
    elites = ranked_by_loss[:elite_count]
    non_elites = ranked_by_loss[elite_count:]
    remaining_slots = max(0, survivor_count - len(elites))

    loss_ranks = {
        id(item): rank
        for rank, item in enumerate(ranked_by_loss)
    }
    ranked_by_relative_improvement = sorted(
        indexed,
        key=lambda item: (
            -(item[2] / max(item[3], 1e-12)),
            item[1],
        ),
    )
    relative_improvement_ranks = {
        id(item): rank
        for rank, item in enumerate(ranked_by_relative_improvement)
    }

    promoted = sorted(
        non_elites,
        key=lambda item: (
            loss_ranks[id(item)] + RELATIVE_WEIGHTING * relative_improvement_ranks[id(item)],
            loss_ranks[id(item)],
            -item[2] / max(item[3], 1e-12),
            item[1],
        ),
    )[:remaining_slots]

    selected = elites + promoted
    return [(permutation, loss) for permutation, loss, _, _ in selected]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)
    device = get_device(args.device)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device={device} seed={args.seed}")
    dataset = load_dataset(device)
    layer_store = load_layers(device)
    even_layers = layer_store.even_layers
    odd_layers = layer_store.odd_layers
    final_layer_id = layer_store.final_layer_id
    print(f"loaded {len(layer_store.layers)} layers and {dataset.inputs.shape[0]} examples")

    state = load_checkpoint(ARTIFACTS_DIR) if args.resume else None
    if state is None:
        population = initialize_search(args.population_size, even_layers, odd_layers, final_layer_id, rng)
        population_losses = evaluate_permutations(population, layer_store, dataset, dataset.sample_full)
        loss_cache = {encode_permutation(permutation): loss for permutation, loss in zip(population, population_losses)}
        best_idx = min(range(len(population)), key=lambda idx: population_losses[idx])
        state = SearchState(
            population=population,
            population_losses=population_losses,
            loss_cache=loss_cache,
            best_permutation=population[best_idx],
            best_loss=population_losses[best_idx],
            generation=0,
            block_size=1,
        )
        print(f"initialized random population of {len(state.population)} valid permutations")
    else:
        if not state.loss_cache:
            state.loss_cache = {encode_permutation(permutation): loss for permutation, loss in zip(state.population, state.population_losses)}
        if len(state.population_losses) != len(state.population):
            state.population_losses = evaluate_permutations(state.population, layer_store, dataset, dataset.sample_full)
            for permutation, loss in zip(state.population, state.population_losses):
                state.loss_cache[encode_permutation(permutation)] = loss
        print(f"resumed generation {state.generation} with population {len(state.population)} and block size {state.block_size}")

    elite_block_exhaustion: dict[bytes, int] = {}
    bubble_search_exhausted: set[bytes] = set()
    full_best_history: list[float] = []
    stagnant_generations = 0
    stop_requested = {"value": False}

    def handle_signal(signum: int, _frame: object) -> None:
        stop_requested["value"] = True
        print(f"signal {signum} received; finishing current generation and checkpointing")

    signal.signal(signal.SIGINT, handle_signal)

    while True:
        generation_start = time.time()
        state.generation += 1
        print(f"\n=== generation {state.generation} max_block_size={state.block_size} ===")
        current_population_keys = {encode_permutation(permutation) for permutation in state.population}
        elite_block_exhaustion = {
            permutation_key: exhausted_block_size
            for permutation_key, exhausted_block_size in elite_block_exhaustion.items()
            if permutation_key in current_population_keys
        }
        elite_count = min(len(state.population), max(1, min(args.survivors, len(state.population)) // 2))
        ranked_current = sorted(
            zip(state.population, state.population_losses),
            key=lambda item: item[1],
        )
        elite_keys = {
            encode_permutation(permutation)
            for permutation, _ in ranked_current[:elite_count]
        }
        bubble_elite_keys = (
            {
                encode_permutation(permutation)
                for permutation, _ in ranked_current[: min(BUBBLE_SORT_ELITES, len(ranked_current))]
            }
            if args.enable_elite_bubble_search
            else set()
        )

        improved_population, improvements, improved_losses = improve_population(
            state.population,
            state.population_losses,
            layer_store,
            dataset,
            state.block_size,
            args,
            rng,
            torch_generator,
            state.loss_cache,
            elite_keys,
            bubble_elite_keys,
            elite_block_exhaustion,
            bubble_search_exhausted,
        )
        prior_losses = list(state.population_losses)
        ranked = sorted(zip(improved_population, improved_losses), key=lambda item: item[1])
        best_perm, best_full_loss = ranked[0]
        if best_full_loss < state.best_loss:
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if stagnant_generations >= STAGNATION_TRIGGER_GENERATIONS:
            improved_population, prior_losses, improved_losses, improvements = apply_stagnation_probe(
                improved_population,
                prior_losses,
                improved_losses,
                improvements,
                layer_store,
                dataset,
                args,
                torch_generator,
            )
            stagnant_generations = 0
            ranked = sorted(zip(improved_population, improved_losses), key=lambda item: item[1])
            best_perm, best_full_loss = ranked[0]

        for permutation, loss in zip(improved_population, improved_losses):
            state.loss_cache[encode_permutation(permutation)] = loss
        ranked_with_improvement = sorted(
            zip(improved_population, improved_losses, improvements),
            key=lambda item: item[1],
        )
        median_loss = ranked[len(ranked) // 2][1]
        avg_improvement = sum(improvements) / max(len(improvements), 1)
        elite_count = min(len(improved_population), max(1, min(args.survivors, len(improved_population)) // 2))
        elite_avg_improvement = (
            sum(item[2] for item in ranked_with_improvement[:elite_count]) / elite_count
            if elite_count > 0
            else 0.0
        )

        if best_full_loss < state.best_loss:
            stagnant_generations = 0
            state.best_permutation = best_perm
            state.best_loss = best_full_loss
            print(f"new best full pred loss {best_full_loss:.8f}")

        full_best_delta_10 = (
            full_best_history[-10] - best_full_loss
            if len(full_best_history) >= 10
            else None
        )
        full_best_delta_25 = (
            full_best_history[-25] - best_full_loss
            if len(full_best_history) >= 25
            else None
        )
        full_best_history.append(best_full_loss)

        delta_10_display = "n/a" if full_best_delta_10 is None else f"{full_best_delta_10:.8f}"
        delta_25_display = "n/a" if full_best_delta_25 is None else f"{full_best_delta_25:.8f}"

        print(
            f"generation {state.generation} summary: full_best={best_full_loss:.8f} "
            f"full_median={median_loss:.8f} "
            f"full_best_delta_10={delta_10_display} "
            f"full_best_delta_25={delta_25_display} "
            f"avg_swap_delta={avg_improvement:.8f} "
            f"elite_avg_swap_delta={elite_avg_improvement:.8f} "
            f"stagnant_generations={stagnant_generations} "
            f"elapsed={time.time() - generation_start:.1f}s"
        )

        survivor_target = min(args.survivors, len(improved_population))
        survivor_pairs = select_survivors(
            improved_population,
            improved_losses,
            improvements,
            prior_losses,
            survivor_target,
        )
        survivors = [perm for perm, _ in survivor_pairs]
        survivor_losses = [loss for _, loss in survivor_pairs]
        print(f"selection: elites={elite_count} blended_slots={max(0, survivor_target - elite_count)}")
        combinations = choose_combination_children(survivors, args.combination_children, final_layer_id, rng)
        cache_keys = set(state.loss_cache)
        survivor_set = {encode_permutation(permutation) for permutation in survivors}
        random_permutations = initialize_search(
            args.random_injections,
            even_layers,
            odd_layers,
            final_layer_id,
            rng,
            excluded=cache_keys - survivor_set,
        )

        next_population: list[tuple[str, ...]] = []
        next_population_losses: list[float] = []
        seen: set[bytes] = set()
        for permutation, loss in zip(survivors, survivor_losses):
            permutation_key = encode_permutation(permutation)
            if permutation_key not in seen and is_valid_permutation(permutation, layer_store):
                seen.add(permutation_key)
                next_population.append(permutation)
                next_population_losses.append(loss)
        for permutation in combinations + random_permutations:
            if len(next_population) >= args.population_size:
                break
            permutation_key = encode_permutation(permutation)
            if permutation_key in state.loss_cache or permutation_key in seen:
                continue
            if is_valid_permutation(permutation, layer_store):
                seen.add(permutation_key)
                next_population.append(permutation)
                next_population_losses.append(float("inf"))

        while len(next_population) < args.population_size:
            candidate = initialize_search(
                1,
                even_layers,
                odd_layers,
                final_layer_id,
                rng,
                excluded=cache_keys | seen,
            )[0]
            candidate_key = encode_permutation(candidate)
            if candidate_key not in seen and candidate_key not in state.loss_cache:
                seen.add(candidate_key)
                next_population.append(candidate)
                next_population_losses.append(float("inf"))

        unknown_indices = [idx for idx, loss in enumerate(next_population_losses) if loss == float("inf")]
        if unknown_indices:
            unknown_permutations = [next_population[idx] for idx in unknown_indices]
            unknown_losses = evaluate_permutations(unknown_permutations, layer_store, dataset, dataset.sample_full)
            for idx, loss in zip(unknown_indices, unknown_losses):
                next_population_losses[idx] = loss
                state.loss_cache[encode_permutation(next_population[idx])] = loss

        state.population = next_population
        state.population_losses = next_population_losses

        if elite_avg_improvement < scaled_small_delta(best_full_loss) and state.block_size < args.max_block_size:
            state.block_size += 1
            print(f"increasing block size to {state.block_size} due to small elite average improvement")

        if best_full_loss <= 0.0:
            print("found zero pred loss on the full dataset")
            save_checkpoint(state, ARTIFACTS_DIR)
            break

        if state.generation % CHECKPOINT_EVERY == 0:
            save_checkpoint(state, ARTIFACTS_DIR)
            print(f"checkpoint saved to {ARTIFACTS_DIR}")

        if stop_requested["value"]:
            save_checkpoint(state, ARTIFACTS_DIR)
            print("stop requested; checkpoint saved and exiting")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("keyboard interrupt received before checkpoint save")
