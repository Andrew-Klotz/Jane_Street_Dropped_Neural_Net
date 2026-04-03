from __future__ import annotations

import random
from typing import Sequence

import torch

from genetic_search_lib.config import ALT_LENGTH, FINAL_INDEX, INPUT_DIM, LayerStore


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
