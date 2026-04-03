# Jane Street Dropped Neural Net

This project reconstructs the correct layer ordering for Jane Street's ["I dropped a neural net"](https://huggingface.co/spaces/jane-street/droppedaneuralnet) puzzle. The puzzle provides all model pieces in randomized order plus historical evaluations, and the goal is to recover the exact permutation of the 97 pieces that rebuilds the original network.

## Results

The solver recovered the exact 97-piece permutation and solved the puzzle, reaching exact zero loss on the full dataset. One successful run took approximately 5 hours on an NVIDIA 3080 and used a GPU-optimized hybrid of genetic and local search.

## Approach

The main search method is a structured genetic/local search hybrid:

- Maintain a population of valid permutations.
- Improve each permutation with local search, primarily through valid block swaps.
- Breed new candidates from survivors using structured combination operations.
- Re-rank candidates on the full dataset and carry forward the strongest population.

The search assumes the known network structure:

- Input is 48-dimensional.
- Residual blocks alternate `48 -> 96` and `96 -> 48`.
- The final layer is the only `48 -> 1` layer.
- All candidate permutations must respect this alternating structure.

Fitness is based on Huber loss against the historical `pred` values. Candidate ranking is done using samples for speed, but permutations are only actually accepted as improvements when they improve on the full dataset.

## Key Engineering Decisions

- Used combinations and swaps as the primary breeding and mutation methods because the expected solution is highly structured.
- Added bubble-style adjacent block scans after observing that neighboring residual block swaps were frequently beneficial.
- Gave top-performing permutations additional search logic, such as bubble-style scans, because elites were more likely to benefit from targeted computation.
- Used survivor logic that keeps elites by absolute quality while also allowing churn through rank-based selection on relative improvement.
- Ranked promising moves on sampled data, but only changed a permutation if the move improved the full-dataset loss.

Over time, several experimental ideas were evaluated:

- Local insertions and 3-block rotations were analyzed and found to produce only small, infrequent improvements, so they did not justify inclusion as primary search operators.
- Large chunk swaps were never observed improving, which fit the observed structure of the problem.
- Stagnation handling mostly turned out to be unnecessary in practice, because structured combinations tended to rise to the top and break plateaus among strong permutations.

## Optimization Methods

The implementation was heavily optimized to make the search practical on GPU:

- Batch candidate evaluation wherever possible.
- Reuse shared prefixes and suffixes across related candidates instead of re-evaluating whole networks from scratch.
- Keep weights loaded on the GPU.
- Use packed tensors plus views and `index_select` to gather the active layers efficiently.

These optimizations were especially important for swap scoring, where many nearby candidates share most of their computation.

## Development Process

The development process was iterative and observation-driven:

- Designed the search architecture and constraints manually.
- Used Codex as an implementation aid for selected components.
- Reviewed generated code for correctness and maintainability.
- Validated behavior through testing, profiling, and direct inspection.
- Refined the system by adding operators and optimizations in response to observed search behavior.

## How To Run

Install requirements:

```bash
python -m pip install -r requirements/base.txt
```

Generate the layer size summary:

```bash
python layer_sizes.py
```

Run the search from scratch:

```bash
python genetic_search.py
```

Resume from the latest checkpoint:

```bash
python genetic_search.py --resume
```

Checkpoints and summaries are written to `artifacts/genetic_search`.

When the solver reaches exact zero loss, it writes the submission-format solution string to `artifacts/genetic_search/solution.txt`.

For the full list of optional search parameters:

```bash
python genetic_search.py --help
```
