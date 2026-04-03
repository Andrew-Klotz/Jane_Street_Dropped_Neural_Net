from __future__ import annotations

import argparse
import random
import signal
import time

import torch

from genetic_search_lib.checkpoints import load_checkpoint, save_checkpoint
from genetic_search_lib.config import (
    ARTIFACTS_DIR,
    BUBBLE_SORT_ELITES,
    CHECKPOINT_EVERY,
    DEFAULT_COMBINATION_CHILDREN,
    DEFAULT_MAX_BLOCK_SIZE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANDOM_INJECTIONS,
    DEFAULT_SURVIVOR_COUNT,
    STAGNATION_TRIGGER_GENERATIONS,
    TOP_K_FULL_4,
    TOP_K_FULL_8,
    TOP_K_INITIAL,
    SearchState,
)
from genetic_search_lib.model import evaluate_permutations, get_device, load_dataset, load_layers
from genetic_search_lib.permutations import (
    choose_combination_children,
    encode_permutation,
    initialize_search,
    is_valid_permutation,
)
from genetic_search_lib.search_ops import (
    apply_stagnation_probe,
    improve_population,
    scaled_small_delta,
    select_survivors,
)
from genetic_search_lib.solution_output import save_solution_file


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
        best_index = min(range(len(improved_population)), key=lambda idx: improved_losses[idx])
        best_perm = improved_population[best_index]
        best_full_loss = evaluate_permutations([best_perm], layer_store, dataset, dataset.sample_full)[0]
        improved_losses[best_index] = best_full_loss
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
            best_index = min(range(len(improved_population)), key=lambda idx: improved_losses[idx])
            best_perm = improved_population[best_index]
            best_full_loss = evaluate_permutations([best_perm], layer_store, dataset, dataset.sample_full)[0]
            improved_losses[best_index] = best_full_loss
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
            solution_path = save_solution_file(best_perm, ARTIFACTS_DIR)
            print(f"solution written to {solution_path}")
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
