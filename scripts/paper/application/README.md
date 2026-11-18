# The weather application

Station-wise postprocessing of 24-hour forecasts of five surface variables
for section 4: marginal models fitted per station and response, then copulas
fitted to their probability integral transforms and scored jointly. The
design combines FLBuchner's (npcc-paper PRs #4 and #6) with the measured
triple below. The code is `npcc.experiments.application`, run through the
`npcc-application` command; the configuration is `configs/application.toml`.

## Data

The anonymized MeteoSwiss station sample of

> Zanetta, F., Nerini, D., Beucler, T. and Liniger, M. A. (2023).
> Physics-constrained deep learning postprocessing of temperature and
> humidity. arXiv:2212.04487.

**The paper must cite it.** It is distributed through the notebook
<https://colab.research.google.com/github/tbeucler/2023_MOOC_ECMWF/blob/main/Physically_constrained_postprocessing.ipynb>,
which downloads `x.nc` and `y.nc` with fixed SHA-256 digests (`DATA_FILES`
in `src/npcc/experiments/application/data.py`). The files live in
`~/npcc-data/meteoswiss_zanetta2023` (`[data] dir`):

```bash
npcc-application fetch --config configs/application.toml               # download and check
npcc-application fetch --config configs/application.toml --verify-only # check files downloaded by hand
```

The server has refused scripted downloads (HTTP 403); the files then come
through a browser. Every stage refuses files whose digest is not the
published one, and records both digests in the run's `provenance.jsonl`.
`npcc.experiments.application.data.write_synthetic` writes synthetic files in
the same layout, for a dry run with `verify = false`.

## Two responses are functions of the other three

Temperature `T` (C), relative humidity `RH` (percent) and surface pressure `p`
(hPa) are measured, stored exactly on a 0.1 grid. The dew point `Td` and the
mixing ratio `w` are computed from them:

- `e = RH / 100 e_s(T)`, with MeteoSwiss's piecewise Magnus formula
  `e_s(t) = c exp(a t / (b + t))`, `a, b, c = 17.368, 238.83, 6.107` hPa at
  and above 0 C and `17.856, 245.52, 6.108` below;
- `Td = e_s^-1(e)`, with the constants of `T`'s sign, as for `e_s(T)`;
- `w = 1000 * 0.622 e / (p - e)` g/kg.

That is how the data was computed: on all 23.8 million rows of `y.nc`,
`(T, RH) -> Td` recovers the stored `Td` with a residual SD of 1.3e-6 C
(float32 rounding); inverting on `Td`'s own branch instead is off by up to
0.24 C where `T` and `Td` straddle 0. A test pins the round trip on the
data; `evaluate` writes the training residuals to
`tables/derivation_residuals.csv`. Since `RH <= 100`, `Td <= T` follows.

So the five responses lie on a three-dimensional surface: their joint
distribution has no density, and a five-variable log score does not compare
forecasts of the data.

## The design

**Margins.** Seven models, each fitted once per station to all five
responses: seasonal EMOS (normal, truncated normal or beta by the response's
support), and LightGBM, TabPFN-quantiles and TabICL, each on all covariates
or on EMOS's (the target's ensemble mean and the day of year), with the
paper's E1 estimators (`[[estimators]]`, pinned to `configs/study.toml` by a
test). RH is fitted on the logit scale, with RH = 100 (or 0) moved to 99.95
(0.05); `w` on the log scale. Each margin gives every case's PIT, the test
cases' log density, and the test quantiles at 199 midpoint levels, rounded
to float32 once predicted, so that the margin's scores, the stored file and
the copula stage see the same numbers. An EMOS margin is also scored by its
closed-form CRPS (`crps_closed_form`), a check of the quantile CRPS that no
skill reads.

**Out-of-fold training PITs.** A margin evaluated at the cases it was fitted
to places them closer to its predictions than new ones: in the triple's
first full run, TabPFN's in-sample training PITs had an SD of 0.26 to 0.27
against 0.289 for uniform ones, LightGBM's down to 0.15. Each margin is
therefore also fitted once per training year without that year's cases, and
the copulas are fitted to those out-of-fold PITs; the test predictions stay
those of the margin fitted on every training case. Units store both PITs,
and `tables/margins_pit_uniformity.csv` compares in-sample, out-of-fold and
test PITs.

**Two copula designs on the same margins.**

- *The triple* (the main design): copulas on `(T, RH, p)`; each draw, mapped
  through the margins' quantiles, gives `Td` and `w` by the formulas above,
  so no draw breaks a physical bound. Its log score is `joint_nll_modeled`.
- *5d* (arms named `<arm>@5d`): copulas on all five responses, as in
  PR #6. Its log score, `joint_nll_5d`, is stored but neither summarized
  nor ranked. An arm and its 5d counterpart share their seeds.

**Arms**, on every margin and in both designs: independence, Gaussian,
boostCopula on all covariates (`r_methods/fit_boostcopula_vine.R`), and the
chain-rule copula with TabPFN, TabICL and LightGBM inside on all covariates
(`npcc-chain-<backend>-all`). On `tabpfn-quantiles-all`, the main table, the
triple also has boostCopula and the TabPFN chain on the day of year, and
the npcc mixtures on all covariates and on the day of year: 88 arms per
station.

- *The chain mixture* is E11's (`npcc.experiments.chain.ChainCopula`):
  `min(mixture_k, d!)` copula densities, each written along a variable
  order drawn per station, five of the six at d = 3. Its density is the
  orders' mean; its draws are split equally across the orders, each drawn
  by its own inverse Rosenblatt transform, so density and draws describe one
  distribution. Each order is also returned as `<arm>-m<k>` and scored by
  its log score alone.
- *The d = 3 mixture* averages the three R-vine structures there are at
  d = 3 with equal weights; member 1 is TLL's Dissmann selection.
- *On the EMOS margins*, whose PITs are not uniform given the covariates, a
  covariate-dependent copula partly recalibrates the margins, and which
  response keeps its margin depends on the order it is drawn in: at station
  0, two fits of one structure with identical log copula densities gave a
  CRPS of T of 0.981 and 1.304 depending on whether T is drawn first. Their
  numbers there read with that caveat.

**Scores**, per arm and test case, each from all 500 draws:

- `joint_nll_modeled` (the triple) or `joint_nll_5d`: the copula's log
  density plus the margins';
- `energy` and `variogram` (p = 1/2) over all five responses, each divided
  by its training standard deviation (in `info.json`, `scales`); and
  `energy_trhp`, `variogram_trhp` over `(T, RH, p)` alone (`[copulas]
  subsets`), which both designs draw;
- `independence@5d` and `gaussian@5d` also store `joint_nll_modeled`, the
  exact log density of `(T, RH, p)` under their own five-variable model
  (`projected_log_score`): independence's is the triple's, the Gaussian's
  the Gaussian copula of its correlation's submatrix;
- `crps_<response>` for all five, from that response's draws;
- `broken_<check>`: how many of the case's draws put `Td` above `T`, RH
  outside [0, 100], `w` below 0, or `p` at or below `e`, and
  `tables/copulas_physical_validity.csv` per station, margin and arm. The
  triple breaks none by construction.

## Outputs

A run folder holds `COMMIT`, `env.txt`, `config.json`, `manifest.json` (the
configuration's signature: a folder refuses a different configuration),
`provenance.jsonl`, `logs/`, `tables/` from `evaluate`, and per station:

- `stations/<NNN>/data/cases.parquet`: the cases and observations;
- `stations/<NNN>/margins/<model>/`: `predictions.parquet` (PITs, log
  density, the `q_*` quantiles), `scores.parquet` and `info.json`;
- `stations/<NNN>/copulas/<model>/<arm>/`: `scores.parquet`, one row per
  result (the arm and each member) and test case, with the scores above;
  `info.json`; and `draws.parquet`, the arm's draws (not its members'),
  readable without npcc:

| column | type | meaning |
|---|---|---|
| `station` | int32 | the station id |
| `model` | string | the marginal model, e.g. `tabpfn-quantiles-all` |
| `run_arm`, `arm` | string | the arm, e.g. `npcc-mixture-tabpfn-quantiles-all` |
| `row` | int64 | the test case's row in `x.nc` and `y.nc`, as in `scores.parquet` |
| `draw` | int32 | the draw, `0` to `n_draws - 1` |
| `obs:air_temperature`, ... | float32 | one column per response, on its own scale (C, percent, hPa, g/kg) |

At 500 draws an arm's `draws.parquet` takes 4.9 MB per station, about
56 GB over 131 stations.

## The summaries (`paired_summary.py`)

Fixed before the run's results were seen; every score lower-is-better,
paired by station, on every station and again without station 106, whose
pressure error steps by about 5 hPa late in its training period
(`pressure_shift.py`). Into `<run>/analysis/`:

1. `pooled_means.csv` and `.md`: every arm's mean of every score over all
   test cases;
2. `paired.csv` and `paired.md`: every arm against `gaussian` on its own
   margins and design (`gaussian@5d` for a 5d arm), and against the anchor,
   `npcc-mixture-tabpfn-quantiles-all` on `tabpfn-quantiles-all`;
3. in the same files, every 5d arm against the same arm of the triple on
   the same margins: energy and variogram on all five responses and on
   `(T, RH, p)`, each response's CRPS, the physical checks' shares, and,
   for independence and Gaussian, the projected log score.

Each difference is the mean over stations of the difference of station
means, with a 95% interval from a block bootstrap over calendar test dates:
500 replicates, each drawing circular blocks of `round(n_dates ** (1/3))`
consecutive dates, the same dates at every station since stations share the
weather on a date, and the stations with replacement
(`scores.date_block_interval`, seed 2026). `paired_by_station.csv` holds
each station's difference.

## Other analyses

Each reads a finished run folder (or the data) and fits nothing:

- `report.py --run <run> [--station 0]`: `reports/report.pdf`, the margins
  (scores and skills against seasonal EMOS per response, PIT histograms and
  coverage, every station and then one) and the copulas (each arm against
  independence on its own margins and design), and `reports/copula_table.csv`,
  the paper's table: every score's pooled mean with its skill, and the gain
  on the main log score;
- `dependence.py binned --run <run>`: Kendall's tau within covariate bins
  (PR #6), on the out-of-fold training PITs of `tabpfn-quantiles-all` and
  `emos-seasonal` (`--models`), per station and pair of the modeled
  responses, in 5 bins (`--bins`) of the day of year and of the ensemble
  mean of the pair's first response (`--covariate`); a bin of fewer than 30
  cases (`--min-pairs`) has no tau. Not on the observations: within a bin
  the margins still move with the other covariates. Into
  `reports/binned_tau/`;
- `dependence.py rolling --run <run> [--station 0] [--window 30]`: rolling
  Kendall's tau of the observations and of the same margins' out-of-fold
  training PITs, into `reports/rolling_tau_<NNN>_<window>/`;
- `eda_physical_consistency.py --data <dir> --out <dir>`: FLBuchner's EDA of
  the raw data (PR #6): coverage, the bounds, and the residuals of RH and
  `w` against the formulas above, also with `e_s(Td)`'s constants picked by
  `Td`'s own sign. On the published files: 297,918 cases at 131 stations;
  the observed `Td` exceeds `T` in 4,464 cases by at most 3.81e-6 C (one
  float32 step); RH and `w` recovered to 5.1e-5 points and 5.2e-6 g/kg;
  with `Td`'s own sign, RH residuals reach 0.085 points;
- `pressure_shift.py --run <run>`: `checks/pressure_shift.csv`, per station,
  the observed pressure less the forecast's ensemble mean on each split and
  its largest step over time. Station 106's steps by -5.09 hPa on
  2020-08-20; no other station's split means differ by more than 0.43 hPa.

## Running

```bash
npcc-application preflight --config configs/application.toml
npcc-application margins   --config configs/application.toml --out <run>
npcc-application copulas   --config configs/application.toml --out <run>
npcc-application evaluate  --config configs/application.toml --out <run>
```

`--stations 0` is the station-0 smoke. `margins` and `copulas` take
`--shard I/K` (every K-th station from the I-th) and `--resume`; `--models`
and `--copulas` narrow the configuration's lists; `copulas --min-free-gb`
stops before a unit, exiting 3, when the disk fills.

`launch.sh` runs it all from a clean worktree at a named commit, with its
provenance (`logs/provenance.txt`, `logs/launcher.log`), in priority order:
the margins (EMOS first, which stores every station's cases; then TabPFN
and TabICL on the GPU beside LightGBM on CPU shards); then the copulas as a
GPU pipeline (the npcc arms with TabPFN or TabICL inside) beside a CPU
pipeline (the rest), each in three phases: (a) `tabpfn-quantiles-all` in the
triple, (b) its 5d arms and `emos-seasonal` in both designs, (c) the rest;
then `evaluate` and `paired_summary.py`. Every copula process resumes, so
`RESUME=1` continues a stopped run, for instance in a second allocation.

**Baobab.** `baobab/` runs it on UNIGE's cluster, in an interactive
allocation started from tmux on the login node: `setup.sh` puts the code
(a git bundle: the repository is private), the Python env and boostCopula's
R library on the GPU node's own disk, since importing torch from BeeGFS
scratch crawls; `start.sh` launches into a run folder on scratch, with
`env.sh` as `launch.sh`'s `SETUP`.

## Runs

- `~/scratch/npcc-runs/application_full_2026-11-18` on Baobab: the run section
  4 reports, `configs/application.toml` at 9f4068f (then named
  `application_full.toml`), on gpu053's A100 with two GPU and two CPU
  processes, started 2026-11-18. Its draws stay on Baobab scratch; `analysis/`,
  `tables/`, the per-case scores, logs, provenance and a SHA-256 manifest of
  the draws come back to `results/`.
- `results/application_consolidated_2026-11-18`: the same design with 200
  draws and fewer arms, at 696acd8 on the local GPU. A fallback only: not
  cited, and not compared with the Baobab run.
- `results/application_triple_2026-11-12`, its smoke and
  `results/application_triple_chain_2026-11-13`: the triple's earlier runs,
  cited by earlier drafts (at dc09eaf, 781314c, and dc09eaf with the chain
  arms at f6c1173; each folder's `provenance.jsonl` has the commits). Their
  dew points used `Td`'s own branch and their copulas in-sample training
  PITs, the two errors the design above fixes. Frozen: later analyses write
  elsewhere.
