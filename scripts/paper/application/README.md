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
uses: Td residual SD *[to fill from the smoke run]*, w residual SD *[to
fill]*. `npcc-application preflight` prints both per station, and
`evaluate` writes them to `tables/derivation_residuals.csv`. The constants
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
  observations and of each margin's training PITs.
