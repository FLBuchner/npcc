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

## E9 — several covariates (§3.3)

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
- Ran from a clean worktree at b74aee4, the merge of the E9 pull request,
  with TAG 2026-11-11. The two runs went at once:
  - `results/study_covariates_2026-11-11`, the npcc arms: 2 GPU shards,
    each with 4 threads on cores 0-7 and `--gpu-mem-fraction 0.45`;
  - `results/study_covariates_2026-11-11_r`, the R competitors: 8
    single-threaded shards pinned to cores 8-15.

  Each run then took its aggregation pass. The analysis the paper cites
  ran later, with the v3.5 arm (below). Each run's `logs/provenance.txt`
  has the exact commands, its `COMMIT` the worktree's state, and its
  `logs/` the scripts that launched them. GPU
  shard 0 failed twice on its first cell with a CUDA error while shard 1
  shared the GPU, then ran alone after shard 1 ended, with `--resume`. A
  cell's seeds depend on the cell alone, so the order changes nothing.
- Analysis: `uv run --no-sync python scripts/paper/e9/analyze.py
  results/study_covariates_<TAG> --with results/study_covariates_<TAG>_r`
  writes `kl.csv` (the main text's `tab:covariates`: KL by scenario x n x
  estimator, averaged over points and families within each rep, with its
  paired difference from TabPFN v3 on every covariate), `kl_by_family.csv`
  (the supplement's `tab:covariates-clayton` and `tab:covariates-gaussian`),
  `iae.csv` (`tab:covariates-iae`), `per_point.csv` and `failures.csv`
  (every fit that failed, each missing from every mean) into the npcc run's
  `analysis/`, and from the R
  scripts' logged lines `fits.csv` and `fits_summary.csv`: per estimator,
  scenario and n, how often the chosen family is the true one, the active
  and inert covariates kept, and how many of boostCopula's stops landed
  within 5% of the cap. It refuses runs that do not pair: a cell on
  different seeds in two runs, or an estimator without every cell of TabPFN
  v3's (the arm on the active covariates aside, under `constant`). A fit
  with a non-finite KL_mc or IAE at any point counts as failed.
- TabPFN v3.5, for the supplement: `configs/e9_covariates_v35.toml` holds
  E1's `tabpfn-quantiles-v3.5-probit` alone, on E9's `[grid]`.
  - Run it as the npcc arms, into `results/study_covariates_<TAG>_v35`.
  - Add that folder to the analysis's `--with`, beside the R run. The
    analysis then pairs it with TabPFN v3 like every other estimator.
  - The main text's TabPFN stays v3, as E1 fixed before E9 ran.
  - Ran from a clean worktree at f8878a7, the merge of the v3.5 and
    selected-structure pull requests, as one GPU process into
    `results/study_covariates_2026-11-11_v35`.
  - The analysis the paper cites then ran from the same worktree at
    f8878a7, into `results/study_covariates_2026-11-11/analysis`:

    ```bash
    uv run --no-sync python scripts/paper/e9/analyze.py \
      results/study_covariates_2026-11-11 \
      --with results/study_covariates_2026-11-11_r \
      results/study_covariates_2026-11-11_v35
    ```

    It replaced the first analysis there, which had the R run alone. Its
    command closes the npcc run's `logs/provenance.txt`. That folder is
    frozen: a later analysis of these runs takes `--out`.
- E9c, E9 at n = 10000:
  - **Why.** It asks whether E9's two gaps at n = 2000 close with more
    data: what the inert covariates cost TabPFN v3 under `interaction`, and
    gamCopula's lead under `gampcc`.
  - **Config.** `configs/e9c_covariates_n10000.toml` and
    `configs/e9c_covariates_n10000_r.toml` are E9's two configs at
    `n = [10000]`, without `constant`, whose inert covariates already cost
    almost nothing at n <= 2000. Every other `[grid]` field and every
    estimator id is E9's, so the covariate points are E9's too.
  - **R arms.** gamCopula's two arms only. A boostCopula fit took about
    1300 s at n = 2000, so at n = 10000 one fit would take hours.
  - **Run and analysis.** As E9's, into `results/study_covariates_n10000_<TAG>`
    and `..._r`.
  - **Ran** on 2026-11-11 from a clean worktree at dd605f2, the tip of the
    configs' pull request, into `results/study_covariates_n10000_2026-11-11`
    and `..._r`, with these lanes:
    - The npcc arms ran as one GPU process (`--workers 1 --resume
      --gpu-mem-fraction 0.9`, 4 threads), which took 2 h 02 min for the 40
      cells.
    - The R arms ran beside it as 8 single-threaded shards with
      `--device cpu`, then the aggregation pass, which took 28 min.
    - The analysis, `analyze.py` on the npcc run with `--with` the R run,
      wrote the npcc run's `analysis/`.
    - A pilot of one cell (`interaction`, Clayton, rep 0) with every arm, in
      `results/e9c_pilot_2026-11-11` and `..._r`, ran first from the same
      commit to size the lanes.
    - Each run's `logs/` holds the launcher, `e9c.sh`, and `provenance.txt`,
      every command rendered.
  - **Cited.** E9c has no table of its own. The paper cites
    `results/study_covariates_n10000_2026-11-11/analysis`, and that folder
    is frozen:
    - `kl.csv` gives the n = 10^4 columns of `tab:covariates` (`gampcc` and
      `interaction`; `constant` is not run, and boostCopula shows "--"),
      and the n = 10,000 numbers in §3.3 and the discussion;
    - `fits_summary.csv` gives the convergence counts in the supplement's
      "Several covariates" subsection: 3 selection fits under `gampcc` that
      did not converge, and none for the oracle;
    - `iae.csv` and `kl_by_family.csv` back that subsection's statement
      that the comparisons at n = 10,000 agree with the KL.

    The n = 10,000 timings in that subsection come from each run's
    `runtime_summary.parquet`, beside `analysis/`: gamCopula selection's
    `fit_time_mean` from the R run, and TabPFN v3's `total_time_mean` from
    the npcc run. The 8 R shards ran single-threaded beside one another and
    beside the GPU process, so these times were taken within the study.

## E2 — vine study (§3.5)

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
- The rerun after E8, the one §3.5 cites: `results/vine_2026-11-10`, from
  b7702e1 on the same config. It ran as 4 shards (`--shard i/4 --resume
  --workers 1 --gpu-mem-fraction 0.15`) started at 19:20 on 2026-11-10,
  then the aggregation pass at 20:50.
  - **No `logs/` or `COMMIT`.** This run has neither. Its commit and flags
    come from the `provenance.jsonl` the runner wrote, one line per
    process, each recording b7702e1 and no uncommitted file.
  - **Analysis.** The command below, on that folder, wrote
    `results/vine_2026-11-10/analysis/`.
- Analysis: `uv run --no-sync python scripts/paper/e2/analyze.py
  results/vine_2026-11-06` writes the table (`vine_table.csv`, in the columns
  of `tab:vine-results`), the figure (`vine_kl_by_x`, PDF and PNG, with the
  plotted values in `vine_kl_by_x.csv`) and the numbers the text quotes
  (`vine_text_numbers.json`) into `results/vine_2026-11-06/analysis/`, and
  prints the table's LaTeX rows. Timings come from E6.
- The selected structure:
  - **What it is.** The `selected` arm is what a user without the true
    structure would fit. The arm's margins are fitted first. Dissmann's
    algorithm with TLL pairs then selects a structure on their probability
    integral transforms, ignoring x: E3d's `tll-ns` call. TabPFN v3 fits the
    pairs on it in a non-simplified context, as for `random`.
  - **What is recorded.** The structure depends on the cell's training data,
    so it varies with n. Each NPCC arm's runtime row records the structure
    it fitted, and the selected arm's also the seconds its selection took.
  - **Config.** `configs/vine_study_selected.toml` is `configs/vine_study.toml`
    with this arm only, so it fits E2's cells: run it into
    `results/vine_<TAG>_selected` with the commands above.
  - **Analysis.** Pass that folder after `--with`. The analysis then writes
    `vine_selected.csv`, one row per (d, n): the arm's joint and copula KL;
    its joint KL paired with the oracle's and the random arm's; how often its
    structure is the true one; and the share of the true first tree and of
    all true edges it recovers. It refuses a run that does not pair: another
    base seed, other evaluation settings or other truths, or another cell
    seed or mean true log-density at any (d, n, rep, x).
  - **Cell data.** Neither `src/npcc/core` nor the truth, data and scoring
    code changed between b7702e1 and this arm. Its pilot on cell d = 5,
    n = 1000, rep 0 reproduced `results/vine_2026-11-10`'s oracle there bit
    for bit.
  - **Ran** from a clean worktree at f8878a7 as one GPU process into
    `results/vine_2026-11-11_selected`. The analysis, with
    `results/vine_2026-11-10` first and `--with` this folder, wrote into
    `--out results/vine_2026-11-11_selected/analysis`, which leaves the
    cited `results/vine_2026-11-10/analysis` as it was. Its `vine_table.csv`
    is byte-identical to the cited one. The run's `logs/provenance.txt` has
    both commands.
  - **Why the first tree differs.** `scripts/paper/e2/tree1_diagnostic.py
    <run>` compares, on each cell's true PITs, the summed pooled |tau| of the
    true first tree, of the selected one, and of the maximum spanning tree.
    It writes `tree1_diagnostic.csv` and `tree1_diagnostic_cells.csv` into
    `<run>/analysis/`, on the CPU in about 30 s at d <= 5. On
    `results/vine_2026-11-11_selected` it ran from 9b26ffa, its command
    closing that run's `logs/provenance.txt`.
  - **Cited.** The paper cites `results/vine_2026-11-11_selected/analysis`,
    and that folder is frozen:
    - its `vine_selected.csv` gives the d = 3 and 5 rows of
      `tab:vine-selected`;
    - its `tree1_diagnostic.csv` gives the tree-1 numbers at d = 3 and 5.
- E2 at d = 10:
  - **Config.** `configs/vine_study_d10.toml` is E2 at `dims = [10]`, with all
    four arms in one run (oracle, random, selected, kde-tll).
  - **Analysis.** Run it on that folder alone. With the selected arm in the
    run itself, the analysis writes `vine_selected.csv` without `--with`. The
    text numbers quoted at d = 3 and 5 are left out, since the run has no
    such cells.
  - **Tree 9.** Tree t's tau band is `[0.05, 0.70 ** t]`, and at t = 9,
    `0.70 ** 9 = 0.040` lies below 0.05. The map onto the band is then
    decreasing, so the single tree-9 edge has its tau in `[0.040, 0.050]`
    with its regime mirrored: near independence either way. Trees 1 to 8
    are as at d <= 5.
  - **Where it runs.** It runs on its own machine, gpu053 (an A100 80GB), as a
    whole experiment: no cell or shard of it runs here. Results there agree
    with this machine's only to about 1e-13, not bit for bit, so it is
    compared with nothing fitted elsewhere. Its `env.txt` records the
    machine.
  - **Ran** on 2026-11-11 on gpu053, from a clean checkout at b6cb9af, the
    merge of the d = 10 pull request, shipped there as a git bundle:
    - The pilot, one cell (n = 1000, rep 0) with all four arms, went into
      `vine_d10_pilot_2026-11-11` and took 13.5 min.
    - The full run went into `vine_d10_2026-11-11` as one process
      (`--workers 1 --resume --gpu-mem-fraction 0.9`, 4 threads) and took
      6 h 14 min for the 40 cells.
    - Each run's `logs/` holds the launcher, `vine_d10.sh`, and
      `provenance.txt`. Both folders were then copied unchanged into
      `results/`.
  - **Analyzed** here on the CPU, from a clean worktree at 9b26ffa.
    Relative to the run's merge, it adds the tree-1 diagnostic. It also
    moves the runner's draw of a cell's training sample into
    `training_sample`, unchanged, so the diagnostic redraws the data every
    arm was fitted on. `analyze.py` and `tree1_diagnostic.py` on
    `results/vine_d10_2026-11-11` wrote its `analysis/`. The tree-1
    diagnostic takes about 80 s at d = 10. The commands close the run's
    `logs/provenance.txt`, beneath gpu053's, with their logs beside them.
  - **Cited.** The paper cites `results/vine_d10_2026-11-11/analysis`, and
    that folder is frozen:
    - `vine_table.csv` gives the d = 10 rows of `tab:vine-results`;
    - `vine_selected.csv` gives the d = 10 rows of `tab:vine-selected`;
    - `tree1_diagnostic.csv` gives the tree-1 numbers at d = 10;
    - `vine_kl_by_x.pdf` is the supplement's Fig. B.3, copied to
      `figures/vine_kl_by_x_d10.pdf`.

    The d = 10 timings come from the run's own `runtime_summary.parquet`,
    beside `analysis/`. They were timed within the study, on gpu053's A100,
    not on an idle machine as E6's were.
- E2b, GAM vines (gamCopula) on E2's cells:
  - **Arms.** Each fits mgcv margins `gam(y_j ~ s(x), method = "REML")`,
    Gaussian, and a GAM vine on their PITs, with x a smooth term in every
    pair and E1b's gamCopula settings, but for two:
    - `gamvine-oracle`: `gamVineCopSelect` on the true structure,
      simplified;
    - `gamvine-selected`: `gamVineStructureSelect`, simplified,
      `treecrit = "tau"`;
    - `gamvine-ns`: `gamVineStructureSelect`, not simplified (Vatter and
      Nagler, 2018).

    The two settings that differ from E1b and E9:
    - `n.iters = 100`, as E9, where E1b ran gamCopula's default of 10;
    - `select.once = TRUE`, gamCopula's default, where E9 passes FALSE. With
      x the only exogenous covariate it hardly matters to the simplified
      arms, and for `gamvine-ns`, FALSE would redo the selection at every
      iteration and multiply its cost.

    **Margins by REML.** mgcv's default smoothing, GCV, undersmoothed 3 of
    the 10 margins on the d = 10 pilot cell (EDF 6 to 8, against linear true
    means). There, against REML, it raised the margins' KL from 0.018 to
    0.050 and gamvine-oracle's joint KL from 0.526 to 0.764, so the margins,
    not the vine, would have decided the comparison. At d = 5 the two agree.

    The joint KL scores the margins and the vine together; the copula KL
    scores the vine at the true PITs, as for every arm.
  - **How it runs.** `r_methods/fit_gamvine.R`, once per arm and cell,
    through `npcc.experiments._external_vine`: it is handed the cell's
    training sample and test points, and the oracle the true structure,
    converted to VineCopula's matrix convention. It checks `gamVinePDF`
    against the product of the fitted pairs' `gamBiCopPDF` densities on
    every fit. Each runtime row records the fitted structure and, in
    `pairs`, every pair copula's family, kind and calibration function. A
    fit that fails is recorded with nan metrics and its `error`.
  - **Three gamCopula 0.0.8 defects, worked around in the script.** In a
    vine, it stores a negatively dependent pair's double family with its
    90- and 270-degree rotations exchanged (`gamVineCopSelect` with its
    default `rotations = TRUE`), and `gamVineStructureSelect` fits each next
    tree on the h-functions of that mirror. And a non-simplified pair reads
    a conditioning variable named `u1` or `u2` as its own argument. The
    script passes `rotations = FALSE` (the same family set), runs the
    structure selection with each pair's h-functions its own, and names the
    copula columns `v1`, ..., `vd`. And with igraph >= 2,
    `gamVineStructureSelect` stops ("Length of new attribute value must be
    1 or ...") whenever a next tree's first candidate pair breaks the
    proximity condition: gamCopula sets the candidates' conditioning sets one
    at a time from none, which older igraph recycled with a warning. It
    stopped the selected and non-simplified fits on 10 of E2's 40 d = 5
    cells in a first run. The script starts those attributes as full-length
    lists, which changes no selected vine (a cell that ran before gives the
    same densities to every digit). `tests/experiments/test_gamvine.py` shows
    each defect's cost or failure.
  - **Configs.** One per E2 run it pairs with, every key but `arms` and the
    script E2's: `configs/vine_study_gamvine.toml` (d = 3, 5),
    `configs/vine_study_gamvine_d10.toml` (d = 10), and their `_n2000`
    versions for E2d's n = 2000 runs. Run each on the CPU from the
    repository root, as single-threaded shards
    (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    npcc-vinestudy --config <config> --device cpu --out <run> --shard i/8
    --resume`, then once without `--shard`), with R, gamCopula, mgcv and
    VineCopula installed.
  - **Analysis.** Pass the run after `--with` to the analysis of the E2 run
    it pairs with. It then writes `vine_gamvine.csv`, one row per (d, n,
    arm): the arm's joint and copula KL, its joint KL paired with the
    oracle's and the random arm's, and its failed fits; and
    `vine_gamvine_families.csv`, the selected pair families per tree.
  - **Pilot.** On 2026-11-11, from cd312e2, into
    `results/e2b_pilot_2026-11-11`: one cell at d = 5 and one at d = 10
    (n = 1000, rep 0), the three arms, R single-threaded, beside other CPU
    jobs. Both cells pair with E2's rows (same seed and true log-density at
    every x). Fit seconds per arm, oracle / selected / ns: 28 / 38 / 94 at
    d = 5, and 135 / 197 / 740 at d = 10; evaluation takes 2 to 8 s. The
    whole d = 10 cell took 18.5 min. `logs/compare_with_e2.txt` has the KLs
    beside E2's arms, and `logs/margins_side_check.txt` what mgcv's default
    smoothing (GCV) costs the margins at d = 10, against REML. The pilot ran
    with GCV margins; the study runs with REML.

- E2d, E2 at n = 2000:
  - **Config.** `configs/vine_study_n2000.toml` (d = 3 and 5) and
    `configs/vine_study_d10_n2000.toml` (d = 10) are E2's configs at
    `n = [2000]`, with all four arms in one run. A cell's truth, test points
    and random structure are drawn from (d, rep) alone, so each cell pairs
    with E2's at n = 500 and 1000 by (d, rep); only the training sample is
    new.
  - **Where it runs.** d = 3 and 5 run here as one GPU process; d = 10 runs
    on gpu053, as a whole experiment, like E2 at d = 10.
  - **Analysis.** `analyze.py` and `tree1_diagnostic.py` on each run alone,
    as at d = 10. The rows for n = 2000 come from these runs and the rows
    for n = 500 and 1000 from E2's.

## E10 — structure mixtures

A non-simplified vine is consistent on any structure, so rather than select
one structure, E10 averages over several. For an equal-weight mixture
`f_bar = (1/K) sum_k f_k`, Jensen's inequality gives
`KL(f || f_bar) <= (1/K) sum_k KL(f || f_k)`, and it holds for the
Monte-Carlo KL point by point.

- **The arm.** `mixture` in the vine study (`vine_runner.py`), with
  `mixture_k = 5` members:
  - The margins are fitted once and shared, as the other NPCC arms fit them.
  - Member 1 is on the `selected` arm's structure: Dissmann's algorithm with
    TLL pairs on the margins' probability integral transforms.
  - Members 2 to K are on pyvinecopulib's weighted random spanning trees of
    the same transforms (`tree_algorithm = "random_weighted"`, weights
    |tau|), each seeded by (d, rep, k). They are diverse, but they favor
    strong pairs: a uniformly random structure is clearly worse than the
    selected one at d = 10 (E3d).
  - Each member is a non-simplified TabPFN v3 vine, fitted and evaluated one
    at a time.
  - The mixture's joint log-density is the margins' plus `logsumexp` of the
    members' copula log-densities less `log K`. Its copula density is the
    members' mean.
  - Every member is also scored on its own, as the arm
    `mixture-member-<k>`, and the mixture's rows carry the members' mean
    KL.
- **Configs.** `configs/vine_study_mixture.toml` (d = 3 and 5, run here)
  and `configs/vine_study_mixture_d10.toml` (d = 10, on gpu053 as a whole
  experiment) hold the mixture arm alone at n = 500, 1000 and 2000. Every
  other key is E2's, so each cell has E2's truth, training data and test
  points, and member 1 is the selected arm's fit there.
- **Analysis.** `scripts/paper/e10/analyze_mixture.py <run> --with <E2
  runs>` writes `vine_mixture.csv`. Per (d, n), it has the mixture's joint
  and copula KL against member 1 and against the members' mean, paired by
  repetition, and against E2's arms on the cells both runs hold. It also
  writes `vine_mixture_checks.json`, which holds Jensen's bound and how far
  member 1 is from the selected arm.
- **Reusing quantile tables.** A pair copula in a vine reads each direction
  twice at the same conditioning rows, for its density and then for its
  h-function. Each `ConditionalMargin` now keeps its last table and returns
  it for a repeat read. `scripts/paper/e10/check_table_reuse.py` checks that
  this changes no TabPFN vine density bitwise, on one E2 cell and one E3d
  sample, and measures the speed-up.

## E3 — comparison with VDC (§3.4)

See `scripts/paper/e3/README.md` for VDC's environment, the weight check and
the command behind every file in `results/e3_vdc/`. It also covers E3d,
E3e (`results/e3e_simplification_2x2/`) and E10 on E3d's truths
(`results/e10_e3d_2026-11-11/`).

## E4 — numerical coherence (§3.6)

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

## The qnorm screen (TabPFN's conditioning feature)

**A screen only.** It asks whether TabPFN v3 fits better when the
copula-scale score it conditions on reaches it as `qnorm(u) = Phi^{-1}(u)`
rather than as `u`. Every TabPFN experiment above hands it `u`. Adopting
`qnorm(u)` would mean rerunning all of them, which is the user's decision,
not this screen's.

- **The setting.** `feature_transform` in the fit controls
  (`FitControlsRosenblattBicop`) and in a config's `[[grid.estimators]]`
  entry. It maps the conditioning score of each Rosenblatt direction (`u` for
  `V | U, X`, `v` for `U | V, X`) after the clamp at `eps`. `"identity"`, the
  default, leaves every fit as before, bit for bit, and enters no estimator
  id. Covariates are never mapped. In a non-simplified vine, the
  conditioning-set values `u_D` reach each pair among its covariates, so
  they stay on the `u` scale: the pair cannot tell them from `x`.
- **The arms.** E1's `tabpfn-quantiles-v3-probit` (v3, probit, softmax
  temperature 1.0, E1's estimator id) and `tabpfn-quantiles-v3-qnorm-probit`,
  the same with `feature_transform = "probit"`, an estimator id of its own.
  Both are in `configs/qnorm_screen.toml`.
- **The cells.** The two arms are paired on identical data:
  - E1's 16 cells at reps 5-9, the reps E7a held out, 80 pairs. The config
    is E1's grid with `first_rep = 5`, so rep 5 is E1's rep 5, and no
    surfaces.
  - E3a's nine copulas, five seeds each, at `n_train` 2000 and 10000: 90
    pairs, from E3a's samples in `results/e3_vdc/e3a`, which the fits only
    read (`e3a_npcc_fit.py --out`). These five seeds are all E3a has; they
    are not held out.
- **The metric.** Each study's pdf KL: E1's Monte Carlo KL averaged over a
  cell's 10 x values, and E3a's KL at its 20,000 test points.
- **The rule**, fixed before any result of the screen was read
  (`scripts/paper/qnorm_screen/report.py`'s `verdict`):
  - per pair, `d = KL(qnorm) - KL(u)`;
  - a stratum is a study at one n: E1 at 200, 500, 1000 and 2000, E3a at
    2000 and 10000;
  - a group's score is the mean of `d` within each of its strata, averaged
    over the strata with equal weight, with SE `sqrt(sum se_s^2) / S`;
  - `qnorm(u)` is adopted only if the score over all pairs is below -2 SE
    (a paired gain beyond 2 SE overall) and no n and no family has a score
    above +2 SE (no loss beyond 2 SE). The n are 200, 500, 1000, 2000 and
    10000. The families are E1's clayton and gaussian and E3a's seven; a
    family in both studies is one group;
  - otherwise `u` is kept. pdf IAE is reported the same way and does not
    decide.

  The differences are absolute: E3a's independence copula has a KL below
  its own Monte Carlo error, so a ratio to it would be noise.
- **Why not E7a's harness.** E7a's and the temperature A/B's scripts
  (`scripts/qt_ab/`) live on the `qt-ab-probit` branch, and replay quantile
  read-outs on cb04041's package, which has no `feature_transform`. The
  screen runs E1's own runner and E3a's own fit script instead, from this
  branch.
- **Run.** From a clean worktree at the pushed tip of the screen's PR, as
  one GPU process at a time (`--workers 1`, 4 threads on cores 0-3, no
  `PYTORCH_CUDA_ALLOC_CONF`):

  ```bash
  WANT=<sha> TAG=<tag> setsid -f nohup \
    bash <worktree>/scripts/paper/qnorm_screen/run.sh > qnorm_screen.out 2>&1
  ```

  It refuses an existing `results/qnorm_screen_<TAG>`, a worktree that is
  not clean or not at `WANT`, a package imported from elsewhere, and a busy
  GPU. It writes `COMMIT`, `env.txt` and `logs/` (the launcher,
  `provenance.txt` with every command, `steps.log`), then `e1/`, `e3a/`, and
  `analysis/` with `pairs.csv`, `groups.csv` and `report.txt`, whose last
  line is the verdict. A step that fails stops the run; its command in
  `provenance.txt` resumes from its own checkpoints.
- **Expected time:** about 1 h on an otherwise idle GPU. E1's cells take
  about 10 s each per arm (E6's clean E3a pdf time, 5.8 s for 20,000
  points, sets the holdout; the grid, h-functions and tau take the rest),
  so 80 cells and two arms are about 27 min. E3a's fits take about 7 s at
  `n_train` = 2000 and 12 s at 10000, so 90 fits and two arms are about
  29 min. If it runs much longer, the baseline arm can be dropped and the
  qnorm arm paired with E1's `results/study_2026-11-10` and E3a's stored
  `tabpfn-quantiles-v3-probit` fits, both from b7702e1, whose
  `src/npcc/core` this branch leaves as it was with the setting off: half
  the time, at the cost of pairing across runs.
