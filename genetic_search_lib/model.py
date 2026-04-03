from __future__ import annotations

import csv
from typing import Sequence

import torch

from genetic_search_lib.config import (
    ALT_LENGTH,
    DATA_DIR,
    HUBER_DELTA,
    HIDDEN_DIM,
    INPUT_DIM,
    PIECES_DIR,
    AffineLayer,
    DatasetBundle,
    LayerStore,
)
from genetic_search_lib.permutations import permutations_to_index_sequences


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
