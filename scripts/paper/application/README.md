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

In the data, relative humidity and the water vapor mixing ratio are computed
from temperature `T`, dew point `Td` (both in degrees Celsius) and surface
pressure `p` (hPa):

- `RH = 100 e_s(Td) / e_s(T)`, the Magnus formula with Sonntag's (1990)
  constants, `e_s(t) = 6.112 exp(17.62 t / (243.12 + t))` hPa. Its largest
  error over all 131 stations is 0.17 RH points.
- `w = 622 e / (p - e)` g/kg with `e = e_s(Td)`. Its log residual has a
  standard deviation of 0.0017.
- `Td` is clipped at `T`: `RH = 100` exactly where `Td = T` exactly, in 4.4%
  of the training rows, at 104 stations; `Td > T` never occurs.

The constants are in `src/npcc/experiments/application/derived.py`.

**Why a five-variable log score is not defined.** The five responses lie on
a three-dimensional surface of five-dimensional space, so their joint
distribution has no density, and a five-dimensional copula times five
margins is a density for a distribution the data cannot come from. Its log
score at an observation measures nothing proper: a model is not rewarded for
putting mass where the data is, since the data has no density to match. On
top of that, `RH = 100` is an atom.

**The triple.** `configs/application_triple.toml` (and its station-0 smoke,
`application_smoke_triple.toml`) model `(T, Td, p)` only:

```toml
[responses]
modeled = ["obs:air_temperature", "obs:dew_point_temperature", "obs:surface_air_pressure"]
derived = ["obs:relative_humidity", "obs:water_vapor_mixing_ratio"]
saturation_nudge = 0.05
```

The margins are fitted to the three, and the copulas to their PITs (d = 3).
`configs/application.toml`, the five-response design, stays the default and
is unchanged; neither list nor the nudge is set there.

**Derived responses.** Each copula draw of `(T, Td, p)`, mapped through the
margins' quantiles, gives a draw of RH and w by the formulas above. A draw
with `Td > T`, which the data never has, is clipped as the data clips:
`min(Td, T)` in both formulas, so RH = 100 there. Each test case's share of
clipped draws is stored (`clipped_share`) and averaged per station, marginal
model and arm in the copula tables. A derived response must have its inputs
modeled; the mixing ratio's are all three, the temperature for the clip.

**Scores of the triple.** Per copula arm and test case:

- `joint_nll_modeled`: the log score of the modeled responses only, the
  copula's log-density plus their three margins'; named so that it cannot
  be mistaken for a five-variable one, with the lists of modeled and
  derived responses in each unit's `info.json`;
- `energy` and `variogram` (p = 1/2) over all five responses, the derived
  ones from the derived draws, each response divided by its training
  standard deviation;
- `crps_<response>` for all five: the one-dimensional energy-score estimator
  on that response's draws, in its own units, so that RH and w get a CRPS.

The margins themselves are scored on the three modeled responses with the
quantile CRPS, as in the default design.

**The saturation atom.** Nothing models it. As FLBuchner's RH nudge moves
an RH of exactly 100 to 99.95, a case of the triple with `Td = T` has its
dew point moved, before its margins see it, to `T - delta(T)`, where the
Magnus RH is `100 - saturation_nudge` (0.05 points). The counts of moved
training and test cases are stored per station and marginal model and
written to `tables/saturation_by_station.csv`; the CRPS and the copula's
energy and variogram scores use the observed dew point.

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
