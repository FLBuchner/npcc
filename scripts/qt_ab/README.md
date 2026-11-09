# The quantile-table read-out A/B

These scripts chose how `QuantileTable` reads a backend's predicted quantiles:
one smoother for every backend, penalized least squares weighted by the
precision of a sample quantile, at bandwidth 0.65
(`npcc.core._quantile_table`). They compare read-outs on E1's cells with the
fitted models held fixed, and check the read-out on exact quantiles where the
truth is known.

All commands run from the repository root. Outputs go under
`results/qt_ab/`, which is not tracked; `QT_AB_RESULTS` moves the single
checks' CSVs elsewhere, and `ab.py` writes where `--out` says. Every row of
every output, CSV or parquet, records the commit it was produced from, and
whether tracked files differed from it.

## The paired A/B on E1's cells

`ab.py` fits each estimator once per E1 cell (`base_seed = 317`), records
every raw quantile row its two margins are asked for while E1's scoring runs,
and scores each read-out arm by replaying those rows through the same scoring.
Arms therefore differ only in the read-out. A backend that fits each level on
its own (LightGBM) or reads every level off one predicted distribution
(TabPFN, TabICLv2, Nori) is recorded once on 81 levels, which contain every
grid compared; XGBoost and CatBoost fit all levels with one loss and are
refitted per grid.

```bash
# The tree backends, logit and probit, n in {200, 500, 2000}, 10 reps: CPU,
# 4 single-threaded processes, about an hour on a shared 16-core machine.
OMP_NUM_THREADS=1 uv run --no-sync python scripts/qt_ab/ab.py \
  --out results/qt_ab/gb --workers 4

# TabPFN v3 under logit, n in {200, 2000}: one GPU process, ~2.3 GB.
uv run --no-sync python scripts/qt_ab/ab.py --out results/qt_ab/tabpfn \
  --estimators tabpfn-quantiles-v3-logit --device cuda --n 200 2000

# TabICLv2 and Nori, the four arms that decide between the two grids.
uv run --no-sync python scripts/qt_ab/ab.py --out results/qt_ab/icl \
  --estimators tabicl-logit nori-logit --device cuda --n 200 2000 \
  --grids m21 m19 --arms 'ma1|current' 'wh0.65|current'
```

The arms (`readouts.py`) cross four smoothers -- none, a 3- or a 7-cell moving
average of the slopes (`ma1`, `ma2`: E1's read-out for the in-context and the
tree backends), and the precision-weighted smoother (`wh0.65`) -- with the
current tails and with modeled-scale ones, on three grids: 21 levels at
`L = 3.25` (E1's), 19 at `L = 2.925` (inside TabICLv2's and Nori's native
0.001-0.999 range) and 41 at `L = 3.25`. `check_readouts.py` checks that the
`ma` arms are E1's read-out and the `wh` arms the library's, bit for bit.

`report.py` prints each arm minus a baseline arm, paired by fitted model, on
the held-out reps 5-9; with `--select` it first ranks every arm on reps 0-4.
`readout_choice.py` writes the one table the paper cites: per backend, n and
arm, the arm minus E1's read-out on the held-out reps, the tree backends
pooled over both transforms; `--reference` lists every sign that flips
against an earlier table.

```bash
uv run --no-sync python scripts/qt_ab/report.py --out results/qt_ab/gb \
  --baseline 'm21:ma2|current' --select --arms 'm21:wh0.65|current' --tau
uv run --no-sync python scripts/qt_ab/report.py --out results/qt_ab/tabpfn \
  --baseline 'm21:ma1|current' --arms 'm21:wh0.65|current' \
  'm19:wh0.65|current' 'm21:ma0|current'
uv run --no-sync python scripts/qt_ab/readout_choice.py \
  --runs results/qt_ab/gb results/qt_ab/tabpfn results/qt_ab/icl \
  --csv results/qt_ab/readout_choice.csv
```

### What it found

pdf KL, mean over all reps, under logit at n = 200 (E1's read-out, then the
weighted smoother; held-out paired differences are in the report):

| backend  | none  | 3 cells | 7 cells | weighted 0.65 |
|----------|-------|---------|---------|---------------|
| LightGBM | 0.310 | 0.194   | 0.143   | 0.124         |
| XGBoost  | 0.219 | 0.141   | 0.119   | 0.102         |
| CatBoost | 0.719 | 0.667   | 0.654   | 0.508         |

- The tree backends fit every level separately, so their tables are noisy
  from level to level and the density turns that noise into spikes: they need
  smoothing, and the weighted smoother beat the 7-cell average for every tree
  backend at n = 200, 500 and 2000, on the held-out reps, with no reversal.
- TabPFN v3 reads every level off one distribution and is nearly indifferent
  to the read-out: against its 3-cell average the weighted smoother moves pdf
  KL by -0.0003 +- 0.0007 at n = 200 (from 0.069) and -0.0029 +- 0.0003 at
  n = 2000 (from 0.012).
- TabICLv2 and Nori (logit, n in {200, 2000}) are close to indifferent, but
  not quite: against their 3-cell average the weighted smoother costs Nori
  +0.003 / +0.002 pdf KL (+3% / +7%) and +0.003 held-out KL (+4% / +15%) at
  n = 200 / 2000, and is mixed for TabICLv2 (pdf KL -0.0027 at n = 2000,
  pdf IAE +0.002 / +0.005).
- 19 levels at `L = 2.925` against 21 at `L = 3.25` makes no measurable
  difference for TabPFN, TabICLv2 or Nori; CatBoost gains a little from it.
- Modeled-scale tails were rejected: they cost CatBoost up to +1.05 pdf KL and
  changed nothing measurable elsewhere.
- XGBoost's flattened tau-hat at strong dependence is mostly the model: no
  smoothing at all lifts it by at most 0.05 of a 0.25 shortfall.

## Checks on exact quantiles and on the backends

| script | output | what it measures |
|--------|--------|------------------|
| `exact_interpolation.py` | `exact_interpolation.csv` | interpolation error on exact copula quantiles: linear in the normal score (the library's) against linear in alpha (TabICLv2's, 3-5 times larger), and the three grids |
| `exact_readout_bias.py` | `exact_readout_bias.csv` | the bias each smoother adds on exact quantiles, and the KL beyond the table under each tail rule |
| `nori_probe.py` | `nori_probe.csv` | Nori on the GPU: its quantile head's dtype (float16), ties and crossings at our levels (none), and the cell its clamped outermost level leaves (half width; density 14-22% high there) |
| `bitwise_dedupe.py` | `bitwise_dedupe_<device>.csv` | predicting each distinct conditioning row once changes nothing for the tree backends, bit for bit; TabPFN's output depends slightly on which rows share a forward pass |
| `tabpfn_chunking.py` | `tabpfn_chunking.csv` | that dependence moves E1's metrics in the fifth decimal |

```bash
uv run --no-sync python scripts/qt_ab/exact_interpolation.py
uv run --no-sync python scripts/qt_ab/bitwise_dedupe.py --device cpu --n 500
```

The outputs the paper's supplement cites were produced at commits of the
`qt-rework` branch, where these scripts were first written:
`nori_probe.csv` and `exact_interpolation.csv` at `ec90142`,
`bitwise_dedupe_cuda.csv` at `bea64ce` and `tabpfn_chunking.csv` at `17ce7f4`.
