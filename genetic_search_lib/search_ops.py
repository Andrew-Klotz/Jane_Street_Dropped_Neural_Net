from __future__ import annotations

import argparse
import random
from typing import Sequence

import torch

from genetic_search_lib.config import (
    ALT_LENGTH,
    INITIAL_SAMPLE_SIZE,
    REFINE_SAMPLE_SIZE,
    RELATIVE_WEIGHTING,
    SMALL_DELTA_RATIO,
    STAGNATION_PROBE_BLOCK_SIZES,
    STAGNATION_PROBE_ELITES,
    TOP_K_FULL_4,
    TOP_K_FULL_8,
    TOP_K_INITIAL,
    DatasetBundle,
    LayerStore,
)
from genetic_search_lib.model import (
    forward_from_boundary,
    forward_shared_tail_from_boundary,
    huber_loss_from_predictions,
    prefix_states_for_permutation,
    random_sample_indices,
)
from genetic_search_lib.permutations import (
    encode_permutation,
    index_sequences_to_permutation,
    permutation_to_index_sequences,
    swap_adjacent_residual_blocks,
)


def scaled_small_delta(loss_scale: float, ratio: float = SMALL_DELTA_RATIO) -> float:
    return max(1e-4, abs(loss_scale) * ratio)


def zero_improvement_meta(baseline_full_loss: float) -> dict[str, float | int | None]:
    return {
        "baseline_loss": baseline_full_loss,
        "best_loss": baseline_full_loss,
        "improvement": 0.0,
        "start_a": None,
        "start_b": None,
        "finalists": 0,
    }


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
