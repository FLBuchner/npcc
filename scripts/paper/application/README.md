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
- `Td = e_s^-1(e)`, on the branch of `Td`'s own sign;
- `w = 1000 * 0.622 e / (p - e)` g/kg.

In the forward direction, `Td -> RH, w`, the data matches to an RH residual
SD of 0.008 points (max 0.085) and a `w` residual SD of 0.00046 g/kg (max
0.0049). The inverse direction, `(T, RH, p) -> Td, w`, which the triple
uses, on each station's training cases in the full run: a Td residual SD of
at most 0.014 C at any station (largest residual 0.16 C) and a `w` residual
SD below 1e-6 g/kg (largest 6e-6). `npcc-application preflight` prints both
per station, and `evaluate` writes them, over the training cases only, to
`tables/derivation_residuals.csv`. The constants
are in `src/npcc/experiments/application/derived.py`, one `Magnus` object
per set; `[responses] magnus` names it (Sonntag's 1990 set is the other).
Since `RH <= 100`, `Td <= T` follows.

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

## Running

```bash
npcc-application preflight --config configs/application.toml
npcc-application margins   --config configs/application.toml --out results/application
npcc-application copulas   --config configs/application.toml --out results/application
npcc-application evaluate  --config configs/application.toml --out results/application
```

The triple design runs the same stages on `configs/application_triple.toml`
into its own folder; `configs/application_smoke_triple.toml` is its station-0
smoke.

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
  `paired_by_station.csv` holds each station's difference;
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
  (its `provenance.txt`), and cited at the paper's 5e9e545.
- `results/application_triple_smoke_2026-11-12`: its station-0 smoke run,
  at 781314c. `checks/order_dependence.csv` (CRPS of T 0.981 against 1.304)
  is cited at the paper's 7b101a7; `checks/README.txt` says how each check
  was made.

Each is frozen: a later analysis writes to a new folder. Neither `copulas
--resume` nor `evaluate` runs into the full run's folder again, since
`evaluate` rewrites `tables/`: the chain arms (the `application-chain`
branch) run in a copy of it checked file by file,
`results/application_triple_chain_2026-11-13`.
