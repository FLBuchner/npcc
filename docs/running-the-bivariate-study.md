# Running the bivariate simulation study

Operational handoff for running `npcc-simstudy` on a large GPU box. It assumes
you have not worked in this repository before, and that the machine is yours
for the duration.

The study fits a Rosenblatt pair copula — a conditional bivariate copula
estimated by two conditional-density regressions — for every combination of
copula family, `tau(x)` regime, sample size and seed in `configs/study.toml`,
and scores each fit against the exact copula on a fixed evaluation grid.

**Read this before starting the full run.** The one thing that is *not* known
is how long it takes; §3 is how you find out, and it is the step most worth
not skipping.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| `uv` | https://docs.astral.sh/uv/. The repo is `uv`-managed; do not `pip install`. |
| A CUDA GPU | Any modern card. The study is not memory-hungry per cell — see §5. |
| `TABPFN_TOKEN` | A TabPFN license token. Most estimators in the grid are TabPFN-backed and will not run without it. |
| Python ≥ 3.11 | `uv` provisions this for you. |

Nothing builds a C++ extension any more: `pyvinecopulib` resolves from PyPI at
`>=1.0.0`, so a sync fetches wheels. If you see a compiler invoked, something
has re-pinned it to git — check `[tool.uv.sources]` in `pyproject.toml`.

## 2. Install and smoke-test

```bash
# Pick the CUDA wheel matching the driver (nvidia-smi shows it):
#   cu126 | cu128 | cu130 | cu132
uv sync --frozen --extra cu128 --extra backends --extra experiments
```

`--frozen` installs the committed `uv.lock` exactly. Keep it. Without it a
fresh box can re-resolve into an old `numba`/`llvmlite` that will not build.

Prove the pipeline end to end before spending GPU hours on it. The test suite
is hermetic — it fakes TabPFN — so this needs no token and no GPU:

```bash
uv run pytest -q tests/experiments/
```

All of it should pass. `tests/experiments/test_cli.py` is a complete study run
in miniature, so if it passes, the cell loop, the checkpointing and the
output tables all work.

## 3. Time one cell before committing to the grid

**The wall-time figure in `configs/study.toml` is explicitly not a budget for
the current grid**, and the comment there says so. The one measurement that
exists was taken on an RTX 2080 Ti over a *smaller* grid: `n_rep = 5`,
`n = [200, 500, 1000]`, 10 backends, giving ~2.06 h per seed. The grid has
since grown on three axes at once — `n_rep = 10`, `n` gained `2000` (the
slowest point), and there are 18 estimators. As configured it is **160 cells
and 5760 fits**, each cell fitting every estimator in both Rosenblatt
directions.

So measure. Copy the config, cut every data axis to one entry, and time it:

```bash
cp configs/study.toml /tmp/probe.toml
# In /tmp/probe.toml set:
#   families = ["clayton"]
#   tau_scenarios = ["linear"]
#   n = [1000]
#   n_rep = 1
# Leave the estimator list alone — its length is what you are measuring.

TABPFN_TOKEN=... time uv run npcc-simstudy \
  --config /tmp/probe.toml --out /tmp/probe --workers 1
```

That is one cell. Multiply by 160 for a first estimate, then correct upward:
cells at `n = 2000` cost more than the `n = 1000` you just timed, and they are
a quarter of the grid. If the projected total is unacceptable, the levers in
descending order of effect are `n_rep`, dropping `n = 2000`, and trimming the
estimator list.

Report the number before launching. Sizing the grid is the user's decision,
not an implementation detail.

## 4. Run it

```bash
TABPFN_TOKEN=<token> CUDA_EXTRA=cu128 scripts/run_study.sh
```

`scripts/run_study.sh` wraps the sync and the run with sensible defaults and
passes `--resume`. Its environment variables (`CONFIG`, `OUT`, `WORKERS`,
`GPU_MEM_FRACTION`) are documented at the top of the script. Re-running the
same command after any interruption continues from the last completed cell.

Note the script's header still describes the old grid's footprint (~1.5 GB
VRAM); treat §5 as the current guidance instead.

## 5. Tuning on a large machine

Three knobs matter, and they behave differently from what you might assume.

**`batch_size`** (in `[grid]` in the config, commented out by default) is the
inner-backend inference chunk. Omitted, it takes a device-aware default —
**2000 on CUDA, sized for an 8 GB laptop card**. On a large accelerator this is
conservative and is the first thing to raise; a bigger chunk means fewer
forward passes. It changes throughput and not results.

**`--workers`** runs whole data cells concurrently in a **thread** pool, not a
process pool. Two consequences:

- It overlaps well where the work releases the GIL — the TabPFN and other
  torch backends — and poorly where it does not, which is the sklearn-style
  CPU backends in the grid (`ngboost`, `xgboost`, `pytabkit`). A many-core box
  does *not* translate into linear speedup for those.
- All workers share one CUDA context and one device, so raise it for GPU
  occupancy rather than expecting it to scale with core count. Each concurrent
  cell also holds its own fitted backends, so memory grows with it.

Start at `--workers 1`, establish a baseline from §3, then raise it and
re-measure. Do not assume.

**`--gpu-mem-fraction`** caps this process's share of VRAM. It exists so a peak
that would starve a display server raises a clean OOM instead of hanging the
machine. On a headless accelerator you can leave it at the script's default.

## 6. Resume semantics — what invalidates a checkpoint

Each cell is checkpointed under `<out>/cells/`, and `<out>/manifest.json`
holds a hash of everything that changes cell *outputs*. On `--resume` the hash
is verified, and a mismatch raises:

> Grid signature changed since the checkpoint in this output directory; use a
> fresh `--out` or drop `--resume`.

This is a feature. Changing `families`, `tau_scenarios`, `n`, `n_rep`,
`normalize`, the grid resolutions, the estimator list or the base seed all
invalidate completed cells, because they change what those cells mean.

**`batch_size` is excluded from that hash, and the exclusion is the point.** It chunks the query
set and leaves each prediction's context alone, so it changes speed and not
results — which means you can re-tune it for this machine partway through a
run and keep every cell already paid for. That is the intended workflow; use
it.

## 7. Outputs

Under `--out`:

| Path | Contents |
|---|---|
| `summary.*` | The headline table: metrics aggregated by `x`. |
| `metrics_by_x.*`, `summary_by_x.*`, `summary_over_x.*` | Aggregations at different levels. |
| `diagnostics.*` | Margin and dependence checks per fit. |
| `quantities.*` | Plot-ready surfaces, for the families in `surface_families` only. |
| `runtime.*` | Per-fit timings. |
| `cells/` | Per-cell checkpoints. Do not delete mid-run. |
| `manifest.json` | The resume guard from §6. |

Format is `parquet` by default and needs `pyarrow`; `--format csv` avoids that.

## 8. Things that will bite

- **Do not run anything else heavy on the machine.** Beyond the obvious
  contention, `runtime.*` is a timing table, so a concurrent job silently
  corrupts numbers you may want to report.
- **`normalize` is `["none"]` and should stay that way unless you mean it.**
  The Sinkhorn projection degenerates (±inf KL) for the sharper parametric
  densities; hardening it is open work. Adding entries is cheap now — the
  study evaluates the whole axis in one call and pays for the backend passes
  once — so cost is no longer the reason to leave it alone. Correctness is.
- **A missing `TABPFN_TOKEN` fails late**, inside the first fit, not at
  startup. Export it before the long run.
- **The estimator list is not the backend list.** Eighteen estimators share
  nine backends across two transforms; several backends need their optional
  extra, which `--extra backends` covers.

## 9. Where the rules live

- `AGENTS.md` — the engineering spec for this repository. Read it before
  changing code, not before running the study.
- `configs/study.toml` — every axis, with the reasoning for each choice inline.
  It is the best single description of what the study actually does.
- `README.md` — the estimator API and the worked examples.

Questions about *what the study means* go to the repository owner. Questions
about *how to run it* should be answerable from this file; if one is not, that
is a bug in this file worth reporting.
