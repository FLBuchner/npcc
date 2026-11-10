# Reproducing the paper's experiments

Every result in the paper and its supplement comes from committed code and
configs in this repository. This file lists, per experiment, what to run and
where the output goes. The experiment plan itself (E1–E8) lives with the paper
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

## Probit only (E8)

The paper presents one transformation of the copula scale, probit: the
normal scores behind TLL in (py)vinecopulib. Every paper config and script
therefore names E1's probit estimators, `<backend>-probit` and TabPFN's
`tabpfn-quantiles-<version>-probit`, with `tabpfn-quantiles-v3-probit` as
the main-text TabPFN. The library keeps logit, its default, and its other
transforms (top-level `README.md`, "Support transforms"); only the paper's
configs leave them out.

The commits and dated result folders named below are those of the runs made
before E8: E1 fitted both transforms, and every other npcc run the logit
estimators only. The commands name the probit ones. E7a's commands stay on
logit: they check out cb04041, whose A/B scripts and labels are those of the
run the supplement cites.

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

- Config: `configs/study.toml` (8 estimators, all probit: the six backends
  and TabPFN's v2.5 and v3.5 releases; 160 cells).
- Run: the sharded commands above with `OUT=results/study_2026-11-05`.
- Analysis: `uv run --no-sync python scripts/paper/e1/analyze.py
  results/study_2026-11-05 --tabpfn <version> --transform probit` writes
  the tables (`overall.csv`, `by_n.csv`, `ranking*.csv`, `margins.csv`) and the
  figures (`accuracy_by_n`, `conditional_tau`, each as PDF and PNG) into
  `results/study_2026-11-05/analysis/`.
- Refitting some estimators after a code change: `npcc-simstudy --config
  configs/study.toml --labels <label> ... --out <fresh folder>` fits only
  those estimators, each exactly as in the full run (same data, same torch
  seed); every shard and the aggregation pass take the same `--labels`.
  `analyze.py` and `paired.py` then take `--replacing <fresh folder>`, whose
  rows replace those labels' rows of the other runs; pass `--out` as well,
  so the original analysis stays as it was.
- Density slices (supplement): `scripts/paper/e1/density_slices.py fit
  --tabpfn <version> --transform probit --out
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
  (v3 at temperature 1.0 and v3.5 at 0.9, probit; E1's cells for n in
  {500, 2000}, reps 0-4) with the sharded commands above and
  `OUT=results/study_2026-11-06_tabpfn_temperature`, from a clean worktree at
  7567717. Each checkpoint's own temperature, which E1 used, is 0.9 for v3
  and 1.0 for v3.5, so the ablation swaps them. Its fits pair with E1's on
  identical data: `scripts/paper/e1/paired.py results/study_2026-11-05 --with
  results/study_2026-11-06_tabpfn_temperature --labels
  tabpfn-quantiles-v3-temp1.0-probit --against tabpfn-quantiles-v3-probit
  --common-cells --out results/study_2026-11-06_tabpfn_temperature/analysis/v3`,
  and likewise v3.5 at 0.9 against `tabpfn-quantiles-v3.5-probit` (into
  `analysis/v3.5`) and against `tabpfn-quantiles-v3-probit` (into
  `analysis/vs_v3`).

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
- CV stops: `scripts/paper/e1/cv_stops.py results/study_2026-11-05_r_smooth_cv`
  writes each cross-validated fit's stop out of the cap (`cv_stops.csv`) and,
  per n, how many stopped within 5% of it (`cv_stops_by_n.csv`), into the
  run's `analysis/`.
- Paired comparison: `scripts/paper/e1/paired.py <run> --with <R run>
  --labels tabpfn-quantiles-v3-probit tabicl-probit best-gb --against gamcopula
  boostcopula-smooth-cv` writes each estimator's pdf KL and IAE minus each
  competitor's, cell by cell, as mean, SE and wins over repetitions (overall,
  by family and by n) into `<run>/analysis/paired.csv`.

## E1c — misspecified scenarios (§3.2)

Two shape families with a closed-form tau(x) of their own, under the
`implied` regime: a Clayton-to-Gumbel mixture with Gumbel weight x (M1) and a
Student t whose degrees of freedom run from 2 to 50 (M2); 80 cells. Both runs
are from a clean worktree at 7567717.

- npcc arms (TabPFN v3, TabICL, Nori, LightGBM, XGBoost, CatBoost):
  `configs/study_misspec.toml` into `results/study_misspec_2026-11-06`, as 2
  GPU shards and the aggregation pass.
- R competitors (gamCopula, boostCopula with CV stopping):
  `configs/study_misspec_r.toml` into `results/study_misspec_2026-11-06_r`, as
  8 single-threaded CPU shards (`--shard I/8 --device cpu`, with
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at 1).
- Analysis: `analyze.py results/study_misspec_2026-11-06 --with
  results/study_misspec_2026-11-06_r`, and `paired.py` with the same runs as
  in E1b.

## E2 — vine study (§3.4)

- Config: `configs/vine_study.toml`: TabPFN v3 with the probit transform;
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
    scripts/paper/e4/coherence.py --labels lightgbm-probit xgboost-probit
    catboost-probit --n 500 2000 --reps 5 --sinkhorn 5 20 50 --device cpu
    --part I/3 --out results/e4_coherence` for I in 0, 1, 2.
  - TabPFN on the GPU, from 7000ae3: the same with `--labels
    tabpfn-quantiles-v3-probit`, no `--device`, `--part I/2` for I in 0, 1,
    and `--out results/e4_coherence/tabpfn`. E1c's scenario code (2b319b7)
    was in the working tree but uncommitted; it reproduces 7000ae3's seeds,
    draws and truths for these cells bit for bit.
- Projection-grid ablation: the same script with `--projection-grid 30 60`
  scores the projection on both grid sizes from one fit per cell, at n = 2000
  on E4's 20 cells. `--sinkhorn 20` into `results/e4_projection_grid/` was
  run for LightGBM on CPU (from 4b83733) and TabPFN v3 on the GPU (from
  7567717); `--sinkhorn 20 50 100` into `results/e4_projection_grid_converge/`
  for TabPFN (from 7a3150a) and LightGBM (from 2cc08f7). Each folder's
  summary adds `projection_grid.csv`, `projection_grid_change.csv` (the
  paired change from unprojected) and `projection_grid_cost.csv`; the
  `proj_*` columns are the margin errors on the projection grid itself.
- Projection cost per distinct conditioning value, on an otherwise idle
  device: the same script with `--cost-only`.
- Summary: `uv run --no-sync python scripts/paper/e4/summarize.py
  results/e4_coherence`, which reads the parts in both folders.

## E6 — clean timings

`bash scripts/paper/e6/run_all.sh` runs every timing, one job at a time, into
`results/e6_timing/`, and refuses to start while another process holds the
GPU or the load average is 2 or more. The jobs, each after an untimed
warm-up:

- bivariate (`scripts/paper/e6/timing.py bivariate`): each Rosenblatt
  direction's fit, density and h-function, and the pair's density, on E1's
  clayton/linear cells at n in {200, 500, 1000, 2000}, 3 reps, for the six
  main-text estimators; the gradient boosters once more on one thread
  (`--gb-threads 1`);
- the R competitors on the same cells, single-threaded, from their scripts'
  own step timers: `configs/e6_timing_gamcopula.toml` (3 reps) and
  `configs/e6_timing_boostcopula_cv.toml` (1 rep; its fit is deterministic
  given the folds);
- vines (`timing.py vine`): the vine runner's per-arm fit and evaluation at d
  in {3, 5}, n in {500, 1000}, 3 reps;
- the projection's cost per distinct conditioning value (E4's
  `coherence.py --cost-only --projection-grid 30 60`) for TabPFN v3 and
  LightGBM at n = 2000;
- the unconditional comparison's timings (`scripts/paper/e3/e6_timing.sh`).

`E6_REUSE_R=<earlier E6 folder>` copies that run's R competitor timings
instead of rerunning them, since no npcc change touches them; each keeps the
commit its own `provenance.jsonl` records, and `logs/provenance.txt` names
the folder they came from.

`scripts/paper/e6/consolidate.py` then writes `timings.csv`, one row per timed
call with the device, CPU, thread settings and commit, and
`timings_summary.csv`, the median seconds and the number of repetitions per
section, method, setting, n, d, k, grid and stage.

## E7a — choice of the read-out (supplement)

The A/B that chose the quantile read-out, and its checks on exact quantiles,
run from cb04041. Their scripts live in that commit's `scripts/qt_ab/`, whose
README describes each one, and every output row records the commit. The
exact-quantile checks and the tree and TabPFN runs come first (2026-11-09);
the TabICLv2 and Nori run and `nori_probe.py` need the whole GPU and follow
E6. Check the commit out to reproduce the four outputs the supplement cites,
into `results/qt_ab_2026-11/`:

```bash
git checkout cb04041
O=results/qt_ab_2026-11
# Exact quantiles, on the CPU in seconds: exact_readout_bias.csv (the bias
# each smoother adds) and exact_interpolation.csv (A.2, the interior).
QT_AB_RESULTS=$O uv run --no-sync python scripts/qt_ab/exact_readout_bias.py
QT_AB_RESULTS=$O uv run --no-sync python scripts/qt_ab/exact_interpolation.py
# The paired A/B on E1's cells: the tree backends on the CPU, the in-context
# learners on the GPU, then readout_choice.csv ("Choice of the read-out").
OMP_NUM_THREADS=1 uv run --no-sync python scripts/qt_ab/ab.py \
  --out $O/gb --workers 4
uv run --no-sync python scripts/qt_ab/ab.py --out $O/tabpfn \
  --estimators tabpfn-quantiles-v3-logit --device cuda --n 200 2000
uv run --no-sync python scripts/qt_ab/ab.py --out $O/icl \
  --estimators tabicl-logit nori-logit --device cuda --n 200 2000 \
  --grids m21 m19 m41 --arms 'ma1|current' 'wh0.65|current'
uv run --no-sync python scripts/qt_ab/readout_choice.py \
  --runs $O/gb $O/tabpfn $O/icl --csv $O/readout_choice.csv
# Nori's quantile head under the adopted read-out: nori_probe.csv (A.2).
QT_AB_RESULTS=$O uv run --no-sync python scripts/qt_ab/nori_probe.py
```

## E7a′ — the read-out under probit (supplement, E8)

The A/B of E7a again, probit only, to fix the read-out for the probit
rerun. It runs from ca4e52c on branch `qt-ab-probit`: cb04041 with two
scripts-only commits. They add the in-context learners' probit labels, a
chosen baseline for `readout_choice.py`, and E7a′'s selection rule
(`report.py --relative-selection`). cb04041's
package has this branch's read-out: its `src` differs only by a comment in
`margin.py` and the `--labels` option.

- **Arms:** on E1's grid of 21 levels (`m21`), with the current tails:
  - no smoothing (`ma0`), which still sorts each row and floors the slopes;
  - the seven-cell moving average (`ma2`);
  - the weighted smoother at bandwidth 0.65 (`wh0.65`, the library's).
- **Backends:** the tree backends at n in {200, 500, 2000}; TabPFN v3,
  TabICLv2 and Nori at n in {200, 2000}.
- **Selection** (fixed before any probit rep 0-4 was read; see
  `report.py`'s `relative_selection`):
  - the score is each pair's relative change in held-out KL against `ma0`,
    averaged per backend and n, then over those rows with equal weight, on
    probit reps 0-4;
  - the lowest score wins, unless a simpler arm (`ma0`, then `ma2`, then
    `wh0.65`) is within 2 SE of it; IAE does not enter;
  - a backend that the pick makes worse than its own best arm by more than
    2 SE and more than 10% is flagged;
  - the pick is then reported on reps 5-9.
- **The held-out KL** is `_common.py`'s: 2000 draws, one per x on E1's
  covariate range. It is not quite the runner's Monte Carlo KL, which takes
  2000 draws at each of the 10 grid values of x, but both estimate the
  x-averaged KL.

Each run needs its own `--out` folder, since `ab.py` skips a cell whose file
exists. Expected time on an idle machine: about 12 min for the trees on four
cores, which can overlap the GPU runs; about 19 min for TabPFN and about
23 min for TabICLv2 and Nori, one after the other.

```bash
git checkout ca4e52c
O=results/qt_ab_probit_2026-11
A=('ma0|current' 'ma2|current' 'wh0.65|current')
# The ma arms are E1's read-out and wh the library's, bit for bit.
uv run --no-sync python scripts/qt_ab/check_readouts.py
OMP_NUM_THREADS=1 uv run --no-sync python scripts/qt_ab/ab.py \
  --out $O/gb --workers 4 --grids m21 --arms "${A[@]}" \
  --estimators lightgbm-probit xgboost-probit catboost-probit
uv run --no-sync python scripts/qt_ab/ab.py --out $O/tabpfn --device cuda \
  --n 200 2000 --grids m21 --arms "${A[@]}" \
  --estimators tabpfn-quantiles-v3-probit
uv run --no-sync python scripts/qt_ab/ab.py --out $O/icl --device cuda \
  --n 200 2000 --grids m21 --arms "${A[@]}" \
  --estimators tabicl-probit nori-probit
# The selection on reps 0-4, one arm for every backend, then the table of
# each arm against ma0 on reps 5-9.
uv run --no-sync python scripts/qt_ab/report.py --out $O/gb $O/tabpfn $O/icl \
  --baseline 'm21:ma0|current' \
  --arms 'm21:ma2|current' 'm21:wh0.65|current' --relative-selection
uv run --no-sync python scripts/qt_ab/readout_choice.py \
  --runs $O/gb $O/tabpfn $O/icl --baseline 'm21:ma0|current' \
  --arms 'm21:ma2|current' 'm21:wh0.65|current' --csv $O/readout_choice.csv
```
