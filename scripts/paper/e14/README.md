# Truncated conditional C-vine versus chain rule

Run from the repository root in an experiment environment with the paper's
TabPFN v3 checkpoint and access configured. For one cell, run

```sh
uv run --no-sync python -m scripts.paper.e14.run \
  --design reversal --n 500 --seed 0 --out results/truncated_cvine_smoke
```

The full run uses both designs, sample sizes 500, 1000, and 2000, and replicate
seeds 0 through 19. It fits 552 conditional regressors per cell.

```sh
uv run --no-sync python -m scripts.paper.e14.run --out results/truncated_cvine
```

Use `--device` to set the Torch device. Completed cells are reused on reruns.
For a cluster run, give each job a distinct zero-based shard index. All jobs
use the same output directory. For example, a 12-job array runs

```sh
uv run --no-sync python -m scripts.paper.e14.run \
  --out results/truncated_cvine --shard "${SLURM_ARRAY_TASK_ID}/12"
```

Use array indices 0 through 11. Shard jobs only write cell checkpoints. A
job with explicit cell selectors can use `--fit-only` for the same behavior.
After all fit jobs finish, run aggregation once.

```sh
uv run --no-sync python -m scripts.paper.e14.run \
  --out results/truncated_cvine --summarize-only
```

Aggregation reads cells marked `DONE` by the shared study runner. Use a
fresh output directory for runs started before that checkpoint format.

Each replicate uses the same structure and order draws across designs and
sample sizes.
For a given design and seed, the same 5,000 balanced test points are reused
across sample sizes. The training regime assignments are independent
Bernoulli draws.

- `summary.csv` averages held-out negative joint log scores, exact-truth
  scores, and their excess across replicates. The rows report overall and
  regime-specific scores. `scores.csv` has per-cell results, including
  counts of nonfinite fitted log densities.
- `paired.csv` reports the five contrasts as left minus right, with standard
  errors across paired replicates. Negative differences favor the left arm.
  `paired_cells.csv` contains the differences used for each contrast.
- `runtime.csv` gives fit time, prediction time, and estimator counts,
  including each ensemble member. Times are in seconds.
- `mass_cells.csv` and `mass_summary.csv` apply E12's truth-proposal
  importance-sampling diagnostic to each RCBicop fit and the mixture.
  The estimate uses the fitted-to-exact density ratio on held-out points.
  A cell passes at an effective sample size of at least 10% of its test
  points, and the summary shows a mass estimate when at least 80% of cells
  pass. Check each regime separately since mass errors can cancel in the
  pooled mean. The ESS threshold assesses estimability. Interpreting excess
  score as KL also requires evidence of unit mass in each regime. The
  truth proposal can miss regions where it places little probability.
- Each cell saves its seeds, matrices, orders, hardware, and settings in
  `result.json`, with per-point test log densities and regimes in
  `log_densities.npz`. `full_true_order` identifies the first member of the
  full vine mixture and supplies both comparisons involving that fit.
