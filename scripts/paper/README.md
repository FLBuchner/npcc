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

## TabPFN at softmax temperature 1.0 (E8)

Every TabPFN release in every paper config runs at softmax temperature 1.0,
no scaling, stated in its `backend_kwargs` as `model_kwargs = {
softmax_temperature = 1.0 }`. Left to the checkpoints, TabPFN 9.2.0 applies
0.9 to v2.5 and v3, whose checkpoints store no temperature, and 1.0 to v3.5,
whose checkpoint stores it, so E1 would compare the releases at different
temperatures. The temperature A/B below tested 1.0 against v3's 0.9 before
the probit rerun.

A label names the release and the transform: `tabpfn-quantiles-v3-probit`
is v3 at 1.0. A temperature other than 1.0 is named in the label, as in the
ablation's `tabpfn-quantiles-v3-temp0.9-probit`. E3 reads its estimators
from E1's config (`scripts/paper/e3/e3_npcc.py`'s `spec`), and E1c, E2, E4,
the density slices and E6 name E1's labels with E1's estimator ids, so all
of them run at 1.0. The runs from before E8 named below used the
checkpoints' temperatures.

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

## The KL the bivariate studies report (E8)

E1, E1b, E1c, the temperature ablation and E4 report a Monte Carlo KL. At
each covariate value of the metric grid, `holdout_n` draws (default 2000, a
`[grid]` key) come from the true conditional copula. They are the same draws
for every estimator of a cell, E4's included: `runner.holdout_points` seeds
them from the cell's seed and the position of the x. The KL is the mean of
`log c - log c_hat` over the draws, with E3's floor of 1e-300 under both
logs. Each pdf row of a run's metrics records it as `KL_mc`, with
`KL_mc_se` (its Monte Carlo standard error) and `KL_mc_floored` (draws at
the floor); E4's parts record `*_KL_mc` beside each KL. IAE stays on the
grid.

The grid KL stays in the tables as `KL` and is not reported. `analyze.py`,
`paired.py`, `r_selection.py` and E4's `summarize.py` report `KL_mc` under
the name KL. They refuse:

- a run without it;
- a NaN from a fitted density that is not finite at some draw, saying so
  (`KL_mc_nonfinite` counts those draws);
- runs scored on different `holdout_n`, whose draws differ (`KL_mc_n`
  records it).

Every dated run below from before E8 therefore has to be rerun before it can
be analyzed, the R competitors (E1b) included: their scripts take the draws
as one more grid, but their fits are not stored. With the same R and
packages, a refit reproduces the stored grid metrics exactly.

On E4's `projection` rows, every density, projected or not, is divided by
its mass at that x before its Monte Carlo KL is scored: a projected density
need not integrate to one, and against an unnormalized density the mean of
`log c - log c_hat` is no divergence. The mass
(`coherence.normal_score_mass`, recorded as `mass_normal_scores`) is the
midpoint rule on 200 x 200 points in normal scores on [-6, 6]. A Clayton
copula at tau 0.7 is unbounded at a corner, and 200 x 200 midpoints in u
overstate its mass by 8e-3; in normal scores the error is 2e-9. `mass`
stays the raw mean over the evaluation grid.

## E1 — bivariate backend study (§3.2)

- Config: `configs/study.toml` (8 estimators, all probit: the six backends
  and TabPFN's v2.5 and v3.5 releases, every TabPFN at softmax temperature
  1.0; 160 cells).
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
- TabPFN temperature ablation (supplement):
  `configs/study_tabpfn_temperature.toml` (v3 and v3.5 at 0.9, against E1's
  1.0, probit; E1's cells for n in {500, 2000}, reps 0-4) with the sharded
  commands above. Its fits pair with E1's on identical data, `E1` and `T`
  being the two runs' folders: `scripts/paper/e1/paired.py $E1 --with $T
  --labels tabpfn-quantiles-v3-temp0.9-probit --against
  tabpfn-quantiles-v3-probit --common-cells --out $T/analysis/v3`, and
  likewise `tabpfn-quantiles-v3.5-temp0.9-probit` against
  `tabpfn-quantiles-v3.5-probit` (into `analysis/v3.5`) and against
  `tabpfn-quantiles-v3-temp0.9-probit` (into `analysis/vs_v3`). The run from
  before E8, `results/study_2026-11-06_tabpfn_temperature` from 7567717,
  swapped the checkpoints' temperatures instead: v3 at 1.0 and v3.5 at 0.9.

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

## E9 — several covariates (§3.2)

The covariates of Vatter and Nagler (2018, JCGS, §4.1): `b1..b5` iid
Bernoulli(1/2), `z1..z5` iid standard normal and `t1..t5` iid uniform, all
independent, so fifteen covariates of three types, binary ones among them.
`npcc.experiments.covariate_scenarios` defines the three scenarios, each
with Clayton and Gaussian copulas:

- **gampcc:** Vatter and Nagler's partially linear predictor, `eta =
  (b1 + b2 - b3)/4 + (z1 - z2 - z3)/4 + s1(t1) + s2(t2) + s3(t3)`, with
  `s1(t) = -1/4 + t/2`, `s2(t) = sin(2 pi t)/4` and `s3(t) = sin(6 pi t)/4`,
  which calibrates Kendall's tau, `tau = tanh(eta/2)`, for either family:
  gamCopula's `tau = TRUE` link, with which Vatter and Nagler's true model
  was built. The Gaussian has `rho = sin(pi tau/2)`; the Clayton is
  gamCopula's double Clayton type I, the Clayton at `theta = 2 tau/(1 - tau)`
  where `tau > 0` and the Clayton rotated by 90 degrees at that of `|tau|`
  below, so Kendall's tau changes sign. pyvinecopulib's rotated Clayton
  equals VineCopula's family 23 at `-theta` to every printed digit, and a
  gamCopula fit whose predictor is the true `eta` predicts the true tau to
  `2e-8` and parameter to `1e-6` relative (gamCopula shrinks tau by `1 -
  1e-8`; `tests/experiments/test_covariate_scenarios.py`); `theta` is
  floored at pyvinecopulib's Clayton bound, `1e-10`. Nine covariates have an
  effect, six none.
- **interaction:** E1's `linear` regime in `t1` where `b1 = 0` and its `sin`
  regime where `b1 = 1`. Two covariates have an effect, and an additive
  model without a `t1 x b1` term is misspecified.
- **constant:** Kendall's tau 0.5; no covariate has an effect. The
  scenario is not named `null`, which pandas reads from a CSV as a missing
  value.

- Configs: `configs/e9_covariates.toml` (npcc) and
  `configs/e9_covariates_r.toml` (the R competitors), on one `[grid]`: the
  two families, the three scenarios, n in {500, 2000}, 10 reps, 120 cells.
- Estimators:
  - npcc on all fifteen covariates, probit: TabPFN v3 at softmax temperature
    1.0, TabICLv2, and LightGBM, the gradient booster with the smallest mean
    KL_mc at n = 2000 on E1's probit rerun (`results/study_2026-11-10`,
    b7702e1: LightGBM 0.0300, CatBoost 0.0373, XGBoost 0.0458, each over 40
    cells). E1's labels and backend settings, so E1's estimator ids.
  - TabPFN v3 on the scenario's active covariates only
    (`active_covariates`; not fitted under constant), and on `(U, V)` alone
    (`without_covariates`).
  - gamCopula as Vatter and Nagler run it, `gamBiCopSelect` with the `b`
    and `z` linear and the `t` smooth, `s(t, k = 10, bs = "cr")`, from
    E1b's family set (Gaussian, Student t, the double Clayton and Gumbel
    with every rotation), with their `select.once = FALSE`
    (`r_methods/fit_gamcopula_select.R`).
  - Both gamCopula arms run `n.iters = 100` Fisher-scoring iterations
    where Vatter and Nagler ran 30: at 30 the oracle can stop unconverged,
    which gamCopula warns is not to be trusted, and a higher cap leaves a
    fit that converges sooner as it is. Each script logs whether its fit
    converged, and the analysis counts those that did not.
  - The oracle, gamCopula with the cell's true family and the scenario's
    true formula (`r_methods/fit_gamcopula_oracle.R`): under `gampcc`, `b1 +
    b2 + b3 + z1 + z2 + z3` and a smooth of each of `t1..t3`; under
    `interaction`, `s(t1, by = b1) + b1` with `b1` a factor; under `constant`,
    the intercept. Each smooth starts at `k = 10` and grows by
    `gamBiCopVarSel`'s own basis rule, ported line for line: double `k`
    while its EDF exceeds `0.8 (k - 1)` and every `k` is below `n / 30`.
    This is the one departure from gampcc's fixed-`k` oracle: at `k = 10`
    the oracle underfits `s3(t) = sin(6 pi t)/4`, where the selection
    arm's doubling does not, so with the same rule the two arms differ only
    in the formula and the family.
  - boostCopula with the `b` and `z` as linear columns and a cubic B-spline
    basis per `t` (`df = 9`, boundary knots `[0, 1]`), stopped by 10-fold
    cross-validation with E1b's cap, `maxit = 5000`
    (`r_methods/fit_boostcopula_smooth_cv_covariates.R`). E1b's stops reach
    the cap's last 5% in 8 of its 160 fits, so a lower cap would change
    fits.
  - Each script receives every covariate, the cell's scenario and its
    family (`npcc.experiments._external`), and logs one line on its fit:
    the family chosen, the covariates kept, and boostCopula's stop.
- Scoring: 20 covariate points per scenario, drawn from the covariate
  distribution with a seed of the scenario's own, so every family, n, rep
  and estimator is scored at the same points (`points.parquet` lists them).
  At each: IAE, ISE and grid KL of the density and both h-functions on E1's
  20 x 20 `uv` grid, and the density's KL_mc at 2000 draws seeded per
  (cell, point).
- Run: the sharded commands above with `npcc-covstudy` for `npcc-simstudy`,
  `--config configs/e9_covariates.toml` and `OUT=results/study_covariates_<TAG>`;
  then `configs/e9_covariates_r.toml` into `results/study_covariates_<TAG>_r`
  with `--device cpu`, single-threaded shards as E1b.
- Analysis: `uv run --no-sync python scripts/paper/e9/analyze.py
  results/study_covariates_<TAG> --with results/study_covariates_<TAG>_r`
  writes `kl.csv` (the main text's table: KL by scenario x n x estimator,
  averaged over points and families within each rep, with its paired
  difference from TabPFN v3 on every covariate), `kl_by_family.csv`,
  `iae.csv`, `per_point.csv` and `failures.csv` (every fit that failed, each
  missing from every mean) into the npcc run's `analysis/`, and from the R
  scripts' logged lines `fits.csv` and `fits_summary.csv`: per estimator,
  scenario and n, how often the chosen family is the true one, the active
  and inert covariates kept, and how many of boostCopula's stops landed
  within 5% of the cap. It refuses runs that do not pair: a cell on
  different seeds in two runs, or an estimator without every cell of TabPFN
  v3's (the arm on the active covariates aside, under `constant`). A fit
  with a non-finite KL_mc or IAE at any point counts as failed.

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

## The TabPFN temperature A/B (supplement, E8)

TabPFN v3 at softmax temperature 1.0 against v3 at its checkpoint's 0.9,
both under the adopted read-out (`m21:wh0.65|current`), on E1's cells for n
in {200, 2000}, reps 0-9, probit. It runs from 42d8fda on `qt-ab-probit`,
two scripts-only commits after ca4e52c: `ab.py`'s
`tabpfn-quantiles-v3-temp1.0-probit`, and the rule below in `report.py
--compare-labels`. `src` is the same at both commits, so E7a′'s
`tabpfn-quantiles-v3-probit` rows are the 0.9 side as they stand.

- **Rule** (fixed before reps 0-4 were read; `report.py`'s
  `relative_labels`):
  - for each n, the mean over pairs (cell, rep) of the held-out KL
    difference, 1.0 minus 0.9, divided by the mean held-out KL at 0.9: a
    ratio of means, so that a pair with a near-zero KL cannot dominate;
  - its SE is that of the mean difference over the same mean;
  - pooled over n with equal weight, SE `sqrt(sum se_n^2) / N`;
  - 1.0 is adopted unless the pooled score is above +2 SE, that is, unless
    it is clearly worse. pdf IAE and reps 5-9 are reported and do not
    decide.
- **Result** (`report_temperature.txt`), as relative changes for 1.0 with
  SE in parentheses:

  | | n = 200 | n = 2000 | pooled |
  |---|---|---|---|
  | held-out KL, reps 0-4 | +8.3% (1.2%) | −4.4% (3.1%) | +1.9% (1.7%) |
  | held-out KL, reps 5-9 | +1.6% (1.8%) | +4.1% (5.9%) | +2.8% (3.1%) |
  | pdf IAE, reps 0-4 | +2.9% (0.6%) | +0.4% (1.5%) | +1.7% (0.8%) |
  | pdf IAE, reps 5-9 | +0.9% (0.9%) | +1.8% (2.6%) | +1.4% (1.4%) |

  The pooled score is 1.2 SE above zero, so 1.0 is adopted. It is not clearly
  worse, but no pooled figure favors it: the case for 1.0 is that it applies
  no scaling to any release, not that it fits better. E1's temperature
  ablation measures the difference on the full grid.

About 13 min on an idle machine.

```bash
git checkout 42d8fda
O=results/qt_ab_probit_2026-11
uv run --no-sync python scripts/qt_ab/ab.py --out $O/tabpfn_temp1.0 \
  --device cuda --n 200 2000 --grids m21 --arms 'wh0.65|current' \
  --estimators tabpfn-quantiles-v3-temp1.0-probit
uv run --no-sync python scripts/qt_ab/report.py --out $O/tabpfn \
  $O/tabpfn_temp1.0 --baseline 'm21:wh0.65|current' --compare-labels \
  tabpfn-quantiles-v3-probit tabpfn-quantiles-v3-temp1.0-probit
```
