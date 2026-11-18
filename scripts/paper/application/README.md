# The weather application

Station-wise postprocessing of 24-hour forecasts of five surface variables:
marginal models fitted per station and response, then copulas fitted to their
probability integral transforms and scored jointly. The experimental design is
FLBuchner's (npcc-paper PR #4); the code is `npcc.experiments.application`, run
through the `npcc-application` command, with the configuration in
`configs/application.toml`.

## Data

The data is the anonymized MeteoSwiss station sample of

> Zanetta, F., Nerini, D., Beucler, T. and Liniger, M. A. (2023).
> Physics-constrained deep learning postprocessing of temperature and
> humidity. arXiv:2212.04487.

**The paper must cite it.** It is distributed through the notebook
<https://colab.research.google.com/github/tbeucler/2023_MOOC_ECMWF/blob/main/Physically_constrained_postprocessing.ipynb>,
which downloads `x.nc` and `y.nc` with fixed SHA-256 digests; the same URLs and
digests are `DATA_FILES` in `src/npcc/experiments/application/data.py`. The
files live outside the repository, in `~/npcc-data/meteoswiss_zanetta2023` by
default (`[data] dir` in the configuration):

```bash
npcc-application fetch --config configs/application.toml
```

downloads both into a temporary file, checks its digest and only then renames
it into place. The server has refused scripted downloads (HTTP 403); `fetch`
then says so, and the two files are downloaded in a browser into that folder
and checked with

```bash
npcc-application fetch --config configs/application.toml --verify-only
```

Every stage refuses files whose digest is not the published one, and each
process records both digests in the run's `provenance.jsonl`. For a dry run on
synthetic files in the same layout
(`npcc.experiments.application.data.write_synthetic`), point `[data] dir` at
them and set `verify = false`.

## Two responses are functions of the other three

Temperature `T` (degrees Celsius), relative humidity `RH` (percent) and
surface pressure `p` (hPa) are measured, and stored exactly on a 0.1 grid.
The dew point `Td` and the water vapor mixing ratio `w` are not on that
grid: they are computed from the other three,

- `e = RH / 100 e_s(T)`, with MeteoSwiss's piecewise Magnus formula
  `e_s(t) = c exp(a t / (b + t))`, `a, b, c = 17.368, 238.83, 6.107` hPa at
  and above 0 C and `17.856, 245.52, 6.108` below, as Zanetta et al. (2023)
  use;
- `Td = e_s^-1(e)`, with the constants of `T`'s sign, as for `e_s(T)`;
- `w = 1000 * 0.622 e / (p - e)` g/kg.

Both constant sets are picked by the sign of the air temperature `T`, also
where `Td` is on the other side of 0. That is how the data was computed:
on all 23.8 million rows of `y.nc`, `(T, RH) -> Td` recovers the stored
`Td` with a residual SD of 1.3e-6 C (largest 8.6e-6, float32 rounding), and
`(T, Td, p) -> RH, w` the stored RH to 5e-5 points and `w` to 7e-6 g/kg.
Inverting on `Td`'s own branch instead leaves residuals of up to 0.24 C on
the 15.9 % of rows where `T` and `Td` straddle 0. The test
`test_the_derivation_reproduces_the_stored_dew_point` pins the round trip
on the data. `npcc-application preflight` prints the residuals per station,
and `evaluate` writes them, over the training cases only, to
`tables/derivation_residuals.csv`. The constants are in
`src/npcc/experiments/application/derived.py`, one `Magnus` object per set;
`[responses] magnus` names it (Sonntag's 1990 set is the other). Since
`RH <= 100`, `Td <= T` follows.

**Why a five-variable log score is not defined.** The five responses lie on
a three-dimensional surface of five-dimensional space, so their joint
distribution has no density, and a five-dimensional copula times five
margins is a density for a distribution the data cannot come from. Its log
score at an observation does not compare forecasts of the data.

**The triple.** `configs/application_triple.toml` (and its station-0 smoke,
`application_smoke_triple.toml`) model the measured triple:

```toml
[responses]
modeled = ["obs:air_temperature", "obs:relative_humidity", "obs:surface_air_pressure"]
derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]
magnus = "meteoswiss"
```

The margins are fitted to the three, each with FLBuchner's choices: RH
keeps its logit transform, its 100 -> 99.95 boundary nudge and its beta
EMOS. The copulas are fitted to their PITs (d = 3). `configs/application.toml`,
the five-response design, is unchanged, though nothing runs it.

**Training PITs.** A copula is fitted to the training cases' PITs and
scored at the test cases'. A margin evaluated at the cases it was fitted to
places them closer to its predictions than new ones, so its in-sample PITs
crowd the middle of [0, 1]. In the full run at dc09eaf, pooled over the 131
stations, TabPFN's in-sample training PITs had an SD of 0.261 to 0.268 per
response, 55.6 to 57.7 % of them in (0.25, 0.75) and 4.9 to 6.1 % in the
outer tenth, against 0.289, 50 % and 10 % for uniform PITs and for its
held-out test PITs; EMOS's SDs were within 0.002 of uniform. With
`[margins] train_pits = "out_of_fold"`, the margin stage also fits each
model once per training year on the cases initialized in the other years
and stores, beside the in-sample `pit` of each training row, its
`pit_out_of_fold` from the model that did not see it; the copula stage
reads the out-of-fold ones. The test predictions are still those of the
model fitted on every training case, with the same seed, so they do not
change. Each fold's seed, case count and seconds are in the unit's
`info.json`, and the stages `fold_fit` and `fold_pit_train` in its
timings. `evaluate` writes `tables/margins_pit_uniformity.csv`: per model,
response and kind of PIT (training in sample, training out of fold, test),
the pooled SD, the shares in (0.25, 0.75) and in the outer tenth, the
Kolmogorov-Smirnov distance to uniform, and the median over stations of
each station's SD.

**Derived responses.** Each copula draw of `(T, RH, p)`, mapped through the
margins' quantiles, gives a draw of `Td` and `w` by the formulas above.

**Scores of the triple.** Per copula arm and test case:

- `joint_nll_modeled`: the log score of the `(T, RH, p)` density, the
  copula's log-density plus the three margins'; named so that it cannot be
  mistaken for a five-variable one, with the lists of modeled and derived
  responses in each unit's `info.json`;
- `energy` and `variogram` (p = 1/2) over all five responses, the derived
  ones from the derived draws, each response divided by its training
  standard deviation;
- `crps_<response>` for all five: the one-dimensional energy-score estimator
  on that response's draws, in its own units.

The marginal stage scores the three modeled margins as before: quantile
CRPS and log score.

**Stored draws.** With `[copulas] save_draws = true`, each copula unit,
`stations/<NNN>/copulas/<model>/<arm>/`, also holds `draws.parquet`: the
draws the arm was scored on, readable with any Parquet reader (pandas,
pyarrow, R's arrow), without npcc. A mixture's members are scored on their
own draws, but only the mixture's are stored. One row per test case and
draw:

| column | type | meaning |
|---|---|---|
| `station` | int32 | the station id |
| `model` | string | the marginal model, e.g. `tabpfn-quantiles-all` |
| `run_arm` | string | the arm as run, e.g. `npcc-mixture-tabpfn-quantiles-all` |
| `arm` | string | the arm, as `run_arm` |
| `row` | int64 | the test case's row in `x.nc` and `y.nc`, as in `scores.parquet` |
| `draw` | int32 | the draw, `0` to `n_draws - 1` |
| `obs:air_temperature`, ... | float32 | one column per response, named as in `y.nc`, on its own scale (C, percent, hPa, g/kg): the modeled ones, then the derived ones |

A chain mixture's orders draw only their share of the mixture's draws; the
mixture's rows hold them all. The per-case scores are in `scores.parquet`
beside it, one row per result (the arm, and each member) and test case:
`logcop`, `joint_nll_modeled`, `energy`, `variogram` and `crps_<response>`
for each of the five. The unit's `info.json` gives, under `scales`, what
the energy and variogram scores divide each response by (its training
standard deviation, `ddof = 0`), and `stations/<NNN>/data/cases.parquet`
the observations. At 363 test cases and 200 draws, an arm takes about
1.45 MB per station before compression, about 190 MB over 131 stations, a
little less compressed, since float noise compresses poorly.
`thin_draws.py --run <run> --arms <arm> ... --keep <k>` keeps the first `k`
draws per case of the arms named, in place, if a run's disk fills: the
scores, computed from every draw before they were stored, do not change;
each thinned unit gets a `draws_thinned.json` and the run's
`logs/draws_thinned.jsonl` a line.

**Physical validity.** Every sampled result's draws are checked, as
FLBuchner's `physics.py` checks them: a dew point above the air temperature
(an RH above 100 %), an RH outside [0, 100], a negative mixing ratio, and a
pressure at or below the vapor pressure `e = e_s(Td)` (with `T`'s
constants), where the mixing ratio's formula breaks down. `scores.parquet`
holds, per test case, how many of its draws break each (`broken_td_above_t`,
`broken_rh_outside_0_100`, `broken_w_negative`, `broken_p_not_above_e`, and
`broken_any` for at least one), and `evaluate` writes
`tables/copulas_physical_validity.csv`: per station, marginal model and
arm, the cases, the draws, and each check's count and share. A check needs
its responses among the drawn ones. For the triple, the first two are zero
by construction, since its RH margin lives on (0, 100) and so `Td <= T`;
the last two would need a pressure drawn below the vapor pressure, which is
at most `e_s(T)`, some tens of hPa. Five responses drawn at once carry no
such guarantee.

**RH = 100.** The cases with RH exactly 100 (or 0) are moved to 99.95 (or
0.05) by the boundary nudge, as in FLBuchner's code. Their counts, per
station, split and marginal model, are stored with each margin unit and
written to `tables/bounds_by_station.csv`.

**The d = 3 mixture.** A 3-d R-vine has exactly three structures, one per
choice of the middle variable of its first tree. Where `mixture_k` reaches
the number of distinct structures on d variables, the npcc mixture arm
averages every one with equal weights instead of drawing weighted random
ones: member 1 is TLL's Dissmann selection, members 2 and 3 the other two
structures in a fixed order, each also scored on its own. At d = 3 with
`mixture_k = 5` the mixture therefore has three members. The random arm
keeps one uniformly random structure per station.

**Which arms run on which margins.** `[copulas] arms_by_margin` narrows the
arms on one marginal model's sets; a model it does not name runs every arm.
The triple's full run fits the npcc mixtures to the TabPFN margins alone, and
the classical arms (independence, Gaussian, boostCopula) to both, so EMOS
with a Gaussian copula is the reference pipeline. An npcc copula conditions
on the covariates. Fitted to PITs that are not uniform given them, as EMOS's
are, it partly recalibrates the margins: its density, the average of its two
Rosenblatt directions, then differs from what its one-directional
inverse-Rosenblatt simulation draws, and which response keeps its margin
depends on the R-vine's order. The station-0 smoke run shows it
(`results/application_triple_smoke_2026-11-12/checks/order_dependence.csv`):
on the EMOS margins, two fits of one structure with identical log copula
densities at every test case give a CRPS of T of 0.981 and 1.304, depending
on whether T is drawn first. The random arm is dropped from the full run,
since at d = 3 it is one of the mixture's members.

**The chain mixture.** `npcc-chain-<backend>-<scope>` is E11's chain rule
(`npcc.experiments.chain.ChainCopula`, `scripts/paper/README.md`, E11) on a
marginal set: `min(mixture_k, d!)` copula densities, each written down
along a variable order, `c(u | x) = prod_k f(u_pi(k) | u_pi(<k), x)`, with
one TabPFN regression per factor. The orders are distinct and drawn
uniformly per station by the vine study's own functions (`chain_orders`, the
station in the repetition's place): five of the six at d = 3. The density
is the orders' mean. The draws are
split equally across the orders (40 of a case's 200 each), and each order
draws its share by its own inverse Rosenblatt transform, from the quantile
tables its density reads, so the mixture's density and its draws describe
one distribution, even where the margins are not uniform given `x`. Each
order is also returned as `<arm>-m<k>` and scored by its log score alone:
its energy, variogram and sample CRPS are `nan` in the tables, and
`paired_summary.py` leaves those comparisons out. At d = 2 the mixture over
both orders is the pair copula's density, the mean of its two Rosenblatt
directions (`tests/experiments/test_chain.py`).

**boostCopula at d = 3.** `r_methods/fit_boostcopula_vine.R` fits any
d >= 2, starting from an initial matrix of the same pattern as FLBuchner's
5 x 5, which it reproduces at d = 5. The matrix only supplies the formulas;
boostCopula selects its own structure and families from that start.

## The consolidated design

`configs/application_consolidated.toml` (and its station-0 smoke,
`application_consolidated_smoke.toml`) is the paper's rerun of section 4. It
consolidates FLBuchner's design in npcc-paper PR #6 -- the comparison of a
copula on all five responses with one on three, the seven margins, the
per-case draws and the physical diagnostics -- with the triple's above:
the measured triple stays the main design, and the five-variable copulas
show how much accuracy a physically possible model gives up.

**Two designs in one run.** `[responses] modeled` and `derived` give the
main design, the triple. `[copulas.designs.<name>]` gives another by its
modeled and derived responses; the consolidated config has one, `5d`, which
models all five and derives none:

```toml
[copulas.designs.5d]
modeled = ["obs:air_temperature", "obs:dew_point_temperature",
  "obs:relative_humidity", "obs:surface_air_pressure",
  "obs:water_vapor_mixing_ratio"]
derived = []
```

An arm named `<arm>@<name>` runs on that design: `gaussian@5d` is the
Gaussian copula on all five responses. The margins are fitted once, to every
response some design models (here all five), and every arm reads its
design's columns of the same stored units. An arm and its other design's
share their seeds, so their comparison sees the same random numbers. A
unit, a table row and a stored draw carry the design in the arm's name,
mixture members included (`<arm>-m<k>@5d`). A design's log score has its own
column, `joint_nll_<name>`, so the triple's `joint_nll_modeled` and the five
responses' `joint_nll_5d` never share one; the latter stays in the files but
is not ranked, since two of the five responses are functions of the other
three and the five have no joint density. A 5d arm is scored like the
triple's on everything drawn: energy and variogram on the five responses,
each divided by the same training standard deviation; each response's
sample CRPS; and the physical checks, which its draws, unlike the triple's,
can break. In the copula tables, a 5d arm's skills and log-score gains are
against `independence@5d` on its margins.

**Float32 quantiles.** `[margins] quantile_dtype = "float32"` rounds each
margin's test quantiles to float32 as soon as they are predicted, before
the marginal scores are computed or the unit is stored, and stores them as
float32 in `predictions.parquet` (the `q_*` columns), half the disk of
float64. The copula stage reads them from the files, so the quantiles the
margin stage scored, the stored ones and the ones the copulas' draws are
mapped through are the same numbers, and a rerun of the copula stage from
the stored files reproduces every score. The default, `"float64"`, is
every earlier run's.

**Arms.** Per marginal model and design, as decided before the run:

| margins | triple | 5d |
|---|---|---|
| emos-seasonal | independence, gaussian, boostcopula-all | independence, gaussian |
| lightgbm-all, lightgbm-emos, tabpfn-quantiles-emos, tabicl-all, tabicl-emos | independence, gaussian, boostcopula-all, npcc-chain-tabpfn-quantiles-all | |
| tabpfn-quantiles-all | independence, gaussian, boostcopula-all, boostcopula-time, npcc-mixture-tabpfn-quantiles-all and -time (each member scored), npcc-chain-tabpfn-quantiles-all and -time | independence, gaussian, npcc-chain-tabpfn-quantiles-all |

The chain stays off the EMOS margins for the reason in section 4: fitted to
PITs that are not uniform given the covariates, a covariate-dependent copula
partly recalibrates the margins, and which response keeps its margin then
depends on the order it is drawn in. Every margin is E1's estimator for its
backend on all five responses (TabPFN v3 with the probit transform at
softmax temperature 1.0; `tabicl-probit`; `lightgbm-probit`), with
out-of-fold training PITs; 200 draws per test case, stored per arm.

**Summaries, fixed before the run.** `paired_summary.py` on the run, every
score lower-is-better, paired by station, on every station and without
station 106:

1. every arm's pooled mean of every score;
2. every arm against `gaussian` on the same margins and design
   (`gaussian@5d` for a 5d arm), and against the anchor,
   `npcc-mixture-tabpfn-quantiles-all` on `tabpfn-quantiles-all`;
3. each 5d arm against the same arm of the triple on the same margins
   (`independence` and `gaussian` on `emos-seasonal` and
   `tabpfn-quantiles-all`; `npcc-chain-tabpfn-quantiles-all` on
   `tabpfn-quantiles-all`), on energy, variogram, each response's sample
   CRPS and each physical check's share of draws, from
   `tables/copulas_physical_validity.csv`; never on the log score.

**Stored draws, and size.** Every score is computed from all 200 draws of
each case before anything is stored, so the paper's numbers do not depend
on what is stored. `[copulas] stored_draws_by_margin` sets, per marginal
model, how many of each case's draws its arms of the main design store, the
first ones; arms of another design, and margins it does not name, store
`stored_draws` (all of them by default); each unit's `info.json` records
its count under `stored_draws`. The consolidated config stores all 200 for
the 13 arms on `tabpfn-quantiles-all` and the 5d arms, and the first 25 for
the triple's 23 arms on the six other margins (independence, gaussian,
boostcopula-all and the chain on all covariates). An arm storing all 200
took 1.9 MB per station in the station-0 smoke, keys included, so the
draws take about 4 GB on disk over 131 stations, against about 9 GB with
all of them; the margins' 35 units per station, with float32 quantiles,
took 16 MB there, about 2.1 GB over 131 stations.

**Launching.** `launch_consolidated.sh` runs the whole design from a clean
worktree at a named commit, into a new folder, with its provenance: the
margins (EMOS first, on the CPU, which stores every station's cases; then
TabPFN and TabICL on the GPU, one process, beside LightGBM on six CPU
shards), the copula stage (the npcc arms on the GPU, one process, beside
the classical arms on six CPU shards), `evaluate` and `paired_summary.py`.
Before each stage it stops if the disk holds less than 2 GB free, or if its
stop file exists. Within the copula stage, `copulas --min-free-gb 2` checks
again before every unit and, below that, exits with status 3 before the
unit is fitted or written, so the same command with `--resume` continues
from it once space is freed.

## The full grid

`configs/application_full.toml` (and its station-0 smoke,
`application_full_smoke.toml`) is the run section 4's numbers come from. It
keeps the consolidated design's margins, responses and estimators, and runs
every copula on every margin in both designs, as FLBuchner's PR #6 does,
with his 500 draws per case, all stored:

- **Arms.** On each of the seven margins and in both designs: independence,
  gaussian, boostcopula-all, and the chain with TabPFN, TabICL and LightGBM
  inside on all covariates (`npcc-chain-{tabpfn-quantiles,tabicl,lightgbm}-all`).
  On tabpfn-quantiles-all, the main table also has boostcopula-time, the
  npcc mixtures on all covariates and on the day of year, and the chain on
  the day of year. 88 arms per station. The npcc copulas run on the EMOS
  margins too, as in PR #6, with the caveat in *The consolidated design*:
  fitted to PITs that are not uniform given the covariates, a
  covariate-dependent copula partly recalibrates the margins.
- **Scores on (T, RH, p).** `[copulas] subsets = { trhp = [...] }` scores
  every sampled result on the triple's modeled responses as well:
  `energy_trhp` and `variogram_trhp`, the full scores' estimators on those
  columns of the same draws, divided by the same training standard
  deviations. Every design draws every response, so the triple and the 5d
  design compare on all five and on these three.
- **Projected log scores.** With `[copulas] projected_log_score = true`,
  `independence@5d` and `gaussian@5d` also store the log density of
  `(T, RH, p)` under their own five-variable model, in `joint_nll_modeled`,
  the triple's column: both projections are exact (a Gaussian copula's onto
  a subset is the Gaussian copula of its correlation's submatrix), so these
  two compare with the triple's log score. The npcc and boostCopula arms
  have no exact projection, and no such score. `joint_nll_5d` is stored but
  neither summarized nor ranked.
- **Closed-form EMOS CRPS.** Every EMOS margin unit's `scores.parquet` also
  has `crps_closed_form`, the CRPS of its normal, truncated normal or beta
  predictive in closed form, on the response's own scale, at the
  observation. The margin tables average it beside `crps`, as a check of the
  quantile CRPS's approximation; no skill reads it, so every margin's CRPSS
  uses one estimator.
- **Size and time.** At 500 draws an arm's `draws.parquet` takes 4.9 MB per
  station (measured at station 0), so about 56 GB over 131 stations. At
  station 0 on one CPU thread, the chain with LightGBM inside took 29 s in
  the triple and 56 s in 5d, and boostCopula in 5d 88 s.

**Launching.** `launch_full.sh` runs it from a clean worktree at a named
commit, in priority order, so the most needed numbers come first and a
second job can finish what a first one's wall time cut short:

1. the margins (EMOS first, which stores every station's cases; then TabPFN
   and TabICL in the GPU processes beside LightGBM on CPU shards);
2. the copula stage, as a GPU pipeline (the npcc arms with TabPFN or TabICL
   inside) and a CPU pipeline (independence, gaussian, boostCopula, and the
   chain with LightGBM inside) side by side, each in three phases:
   (a) tabpfn-quantiles-all in the triple, the main table; (b) its 5d arms,
   and emos-seasonal in both designs, the 5d comparison; (c) the rest. Every
   copula process runs with `--resume`, so a phase skips what an earlier one
   stored;
3. `evaluate` and `paired_summary.py`.

`RESUME=1` continues a run folder a stopped launch left, every stage with
`--resume`; `SETUP` names a file sourced first, such as a cluster's module
loads for R; `GPU_SHARDS` and `SHARDS` set how many GPU and CPU processes run
side by side. The stop file and the disk guards are `launch_consolidated.sh`'s.

## Running

```bash
npcc-application preflight --config configs/application.toml
npcc-application margins   --config configs/application.toml --out results/application
npcc-application copulas   --config configs/application.toml --out results/application
npcc-application evaluate  --config configs/application.toml --out results/application
```

The triple design runs the same stages on `configs/application_triple.toml`
into its own folder; `configs/application_smoke_triple.toml` is its station-0
smoke. The consolidated design runs them on
`configs/application_consolidated.toml`, with
`configs/application_consolidated_smoke.toml` its station-0 smoke.

`margins` and `copulas` take `--shard I/K` (every K-th station from the I-th,
one process each) and `--resume`; `--stations`, `--models` and `--copulas`
narrow the configuration's lists. `configs/application_smoke.toml` runs the
whole design, every marginal set and every copula arm, on station 0.

## Reports

Each reads a run folder that `evaluate` has written and fits nothing:

- `report_margins.py --run <run>`: `reports/marginal_report.pdf`, the scores
  and skills against seasonal EMOS per response, every station and then one,
  with PIT histograms, coverage and timings;
- `report_copulas.py --run <run>`: `reports/copula_report.pdf`, each copula
  arm against independence on its own margins, and `reports/copula_table.csv`,
  the paper's table;
- `plot_rolling_tau.py --run <run> --station 0`: rolling Kendall's tau of the
  observations and of each margin's training PITs;
- `paired_summary.py --run <run>`: the triple's primary summaries, fixed
  before its full run's results were seen, in `<run>/analysis/`.
  `pooled_means.csv` and `.md` hold each arm's mean score over all test
  cases, one table per score. `paired.csv` holds each arm against the
  Gaussian copula on its own margins and against the npcc mixture on all
  covariates on the TabPFN margins: the mean over stations of the
  difference of station means (negative favors the arm), its standard error
  across stations, and the stations where the arm is lower, higher or level;
  on every station and again without station 106, whose pressure error
  steps by about 5 hPa late in its training period (`pressure_shift.py`).
  With arms of another design (*The consolidated design*), each is compared
  with `gaussian` of its own design, and also with the same arm of the main
  design on its margins, on the scores both draw and the physical checks'
  shares. `paired_by_station.csv` holds each station's difference;
- `pressure_shift.py --run <run>`: `<run>/checks/pressure_shift.csv`, per
  station, the observed surface pressure less the forecast's ensemble mean:
  its mean on each split, and the largest single step in it over time, dated.
  In the full run, station 106's error steps from +4.81 to -0.28 hPa on
  2020-08-20 (-5.09 hPa), so 128 of its 1430 training cases and all 363 test
  cases follow the step; its split means differ by -4.58 hPa, and no other
  station's by more than 0.43 hPa.

## Runs

- `results/application_triple_2026-11-12`: the triple's full run
  (`configs/application_triple.toml`), at dc09eaf, on the main machine's
  GPU (an NVIDIA GRID A100D-20C, `env.txt`) with the classical arms on six
  CPU shards; 131 stations, finished 2026-11-13 02:51. Commands are in
  `logs/provenance.txt`. `analysis/` is `paired_summary.py`'s, at dc09eaf.
  Cited at the paper's 7b101a7: `analysis/paired.csv`,
  `analysis/pooled_means.csv`, `tables/margins_pooled.csv`,
  `margins_by_station.csv`, `bounds_by_station.csv`,
  `derivation_residuals.csv` and `copulas_timings.csv`, and `env.txt`.
  `checks/pressure_shift.csv` is `pressure_shift.py`'s, added at b27acaa
  (its `provenance.txt`), and cited at the paper's 5e9e545. Its code
  inverted `Td` on `Td`'s own branch, so its derived `Td` draws and its
  `derivation_residuals.csv` (a Td residual SD of at most 0.014 C at any
  station, largest residual 0.16 C) carry that convention's error, which
  the data's own convention (above) removes. Its copulas were fitted to
  in-sample training PITs (*Training PITs*, above).
- `results/application_triple_smoke_2026-11-12`: its station-0 smoke run,
  at 781314c. `checks/order_dependence.csv` (CRPS of T 0.981 against 1.304)
  is cited at the paper's 7b101a7; `checks/README.txt` says how each check
  was made.

Each is frozen: a later analysis writes to a new folder. Neither `copulas
--resume` nor `evaluate` runs into the full run's folder again, since
`evaluate` rewrites `tables/`: the chain arms (the `application-chain`
branch) run in a copy of it checked file by file,
`results/application_triple_chain_2026-11-13`.
