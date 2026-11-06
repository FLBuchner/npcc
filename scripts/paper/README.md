# Reproducing the paper's experiments

Every result in the paper and its supplement comes from committed code and
configs in this repository. This file lists, per experiment, what to run and
where the output goes. The experiment plan itself (E1–E7) lives with the paper
(`paper/jcgs/experiments.md` in the paper repository).

All commands run from the repository root, with the environment synced as

```bash
uv sync --frozen --extra cu130 --extra backends --extra experiments
```

(pick the CUDA flavor that matches the driver). Results are written under
`results/`, which is not tracked. This file names the commit behind each
reported run. The E3 and E4 scripts also write it into every row, and from
22e0da2 on, every process of a study run appends its commit, uncommitted
files and options to `provenance.jsonl` in the run's folder.

## Running a study on several processes

`npcc-simstudy` runs cells as threads by default, which does not parallelize:
the cells hold the interpreter lock for much of their work. Split the cells
over processes instead, then aggregate once:

```bash
OUT=results/<run>
for i in 0 1 2 3; do
  OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
    uv run --no-sync npcc-simstudy --config <config> --out $OUT \
    --workers 1 --resume --shard $i/4 --gpu-mem-fraction 0.9 &
done; wait
uv run --no-sync npcc-simstudy --config <config> --out $OUT --resume
```

Seeds depend only on each cell, so a sharded run reproduces a single-process
run exactly (`tests/experiments/test_cli.py`). Four processes fit a 20 GB GPU
with TabPFN 3.5 at n = 2,000; wall-clock times from a sharded run are
contaminated by the concurrency and are not reported (see E6).

## E1 — bivariate backend study (§3.2)

- Config: `configs/study.toml` (16 estimators; 160 cells).
- Run: the sharded commands above with `OUT=results/study_2026-11-05`.
- Analysis: `uv run --no-sync python scripts/paper/e1/analyze.py
  results/study_2026-11-05 --tabpfn <version> --transform <transform>` writes
  the tables (`overall.csv`, `by_n.csv`, `ranking*.csv`, `margins.csv`) and the
  figures (`accuracy_by_n`, `conditional_tau`, each as PDF and PNG) into
  `results/study_2026-11-05/analysis/`.
- Density slices (supplement): `scripts/paper/e1/density_slices.py fit
  --tabpfn <version> --transform <transform> --out
  results/study_2026-11-05/analysis/slices` (add `--part I/K` to split), then
  `... density_slices.py plot --out ...`. The refits use E1's cell and
  estimator seeds, so the models equal E1's up to GPU nondeterminism. Every
  row records its commit: 7000ae3 for parts 0–2, d18480b for part 3, rerun
  after a CUDA launch failure (nothing E1 runs changed in between).
- Code: the four shards ran once each from 2f8fa94, and the aggregation pass
  from 7000ae3 only read their checkpoints. The density metrics are
  unprojected. `projection_summary` predates the midpoint Sinkhorn projection
  (a26262c), so the projection results reported come from E4.
- TabPFN temperature ablation (supplement): `configs/study_tabpfn_temperature.toml`
  (v3 at temperature 1.0 and v3.5 at 0.9, logit; E1's cells for n in
  {500, 2000}, reps 0-4) with the sharded commands above and
  `OUT=results/study_2026-11-06_tabpfn_temperature`; its fits pair with E1's
  on identical data, so `analyze.py results/study_2026-11-05 --with
  results/study_2026-11-06_tabpfn_temperature` puts both in one table.

## E1b — established competitors on E1's cells (§3.2)

R 4.6.2 with VineCopula 2.6.1 and mgcv 1.9-4; boostCopula 0.1.0 at commit
746b191 of jobstdavid/boostCopula; gamCopula 0.0-8 built from commit e5561a8
of tvatter/gamCopula (PR #20, which passes a fitted t copula's second
parameter to its density and distribution). Each config's `[grid]` table is
E1's, so cells, seeds and truths are E1's, and each script's SHA-256 enters
its estimator id, so a run's signature pins the script it ran. All runs use
the sharded commands above with `--device cpu`.

- `configs/study_r.toml` into `results/study_2026-11-05_r`, from 349ec08:
  gamCopula, and boostCopula linear in x.
- `configs/study_r_smooth_cv.toml` into `results/study_2026-11-05_r_smooth_cv`,
  from d10182b: boostCopula on a cubic B-spline basis of x, stopped by
  10-fold cross-validation. This is the boostCopula arm reported. It ran as
  8 single-threaded shards (`--shard I/8`, with `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at 1).
- `configs/study_r_smooth.toml` into `results/study_2026-11-05_r_smooth`,
  from b8f26dd: the same basis stopped by AIC, kept for the record. It stops
  at maxit = 500 in 63 of 160 fits.
- Analysis: `analyze.py results/study_2026-11-05 --with
  results/study_2026-11-05_r results/study_2026-11-05_r_smooth_cv` adds the
  `*_with_competitors` figures, and `scripts/paper/e1/r_selection.py <run>`
  joins each R fit's family, formula and EDF with its accuracy into
  `<run>/analysis/r_selection.csv`.
- Paired comparison: `scripts/paper/e1/paired.py <run> --with <R run>
  --labels tabpfn-quantiles-v3-logit tabicl-logit best-gb --against gamcopula
  boostcopula-smooth-cv` writes each estimator's pdf KL and IAE minus each
  competitor's, cell by cell, as mean, SE and wins over repetitions (overall,
  by family and by n) into `<run>/analysis/paired.csv`.

## E2 — vine study (§3.4)

- Config: `configs/vine_study.toml`: TabPFN v3 with the logit transform;
  oracle, random and TLL arms; d in {3, 5}, n in {500, 1000}, 20 reps; 80
  cells.
- Run from a clean worktree at d10182b into `results/vine_2026-11-06`, as 4
  shards of `npcc-vinestudy` with the flags above on two GPU slots, then the
  aggregation pass; `logs/provenance.txt` has the exact commands. A shard
  that ran out of GPU memory was resumed, which reuses its finished cells: a
  cell's seeds depend on the cell alone. A d = 5 shard peaks at about 7 GB,
  since every fitted margin held its own TabPFN model on the GPU before
  f4f17b5.
- Analysis: `uv run --no-sync python scripts/paper/e2/analyze.py
  results/vine_2026-11-06` writes the table (`vine_table.csv`, in the columns
  of `tab:vine-results`), the figure (`vine_kl_by_x`, PDF and PNG, with the
  plotted values in `vine_kl_by_x.csv`) and the numbers the text quotes
  (`vine_text_numbers.json`) into `results/vine_2026-11-06/analysis/`, and
  prints the table's LaTeX rows. Timings come from E6.

## E3 — comparison with VDC (§3.3)

See `scripts/paper/e3/README.md` for VDC's environment, the weight check and
the command behind every file in `results/e3_vdc/`.

## E4 — numerical coherence (§3.5)

- Accuracy and coherence, as two runs that differ only in estimators and
  device; every row records the commit it ran from.
  - The gradient boosters on CPU, from 7909ed7: `uv run --no-sync python
    scripts/paper/e4/coherence.py --labels lightgbm-logit xgboost-logit
    catboost-logit --n 500 2000 --reps 5 --sinkhorn 5 20 50 --device cpu
    --part I/3 --out results/e4_coherence` for I in 0, 1, 2.
  - TabPFN on the GPU, from 7000ae3: the same with `--labels
    tabpfn-quantiles-v3-logit`, no `--device`, `--part I/2` for I in 0, 1,
    and `--out results/e4_coherence/tabpfn`. E1c's scenario code (2b319b7)
    was in the working tree but uncommitted; it reproduces 7000ae3's seeds,
    draws and truths for these cells bit for bit.
- Projection cost per distinct conditioning value, on an otherwise idle
  device: the same script with `--cost-only`.
- Summary: `uv run --no-sync python scripts/paper/e4/summarize.py
  results/e4_coherence`, which reads the parts in both folders.

## E6 — clean timings

With nothing else running on the machine, `uv run --no-sync python
scripts/paper/e6/timing.py bivariate --out results/e6_timing` times each
Rosenblatt direction's fit, density and h-function, and the pair's density, on
E1's evaluation grid, for the six main-text estimators at n in {200, 500,
1000, 2000}, 3 reps each, after one untimed warm-up per estimator. `...
timing.py vine --out results/e6_timing` takes the vine runner's per-arm fit
and evaluation times at d in {3, 5}, n in {500, 1000}, after one untimed
cell. Every row records the device, where the estimator computes, the thread
settings and the commit. The projection cost per distinct conditioning value
(E4, `--cost-only`) runs in the same idle window.

## Still to be filled in

E1c (the misspecified scenarios) and the projection-grid ablation are added
here when their runs are final.
