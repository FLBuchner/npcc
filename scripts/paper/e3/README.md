# E3 — comparison with VDC (§3.4)

The scripts here compare NPCC with VDC (Safaai 2026, the vine denoising
copula; <https://github.com/KempnerInstitute/vine-denoising-copula>) on VDC's
own published settings. They write everything under `results/e3_vdc/`, which
is not tracked. This file says how to build VDC's environment, how to check
its weights, and which command produces each output file.

- **E3a**: unconditional pair copulas, VDC's Table 1 cases, at
  `n_train` = 2,000 and 10,000.
- **E3b**: VDC's non-simplified stress test, the copula of `(U2, U3)`
  switching at `U1 = 0.5`.
- **E3c**: simplified vines with d ∈ {5, 10} on the true structure, with
  every method in one vine machinery. There are two designs: `decay`, the
  primary one, where Kendall's tau shrinks by 0.7 per tree; and `strong`, a
  stress variant with every tree as strong as the first.
- **E3d**: vines on the same pair copulas and designs, but on a random
  R-vine structure that no arm is told. The competitors select a structure
  as a user would, and NPCC fits a non-simplified vine on a random structure
  or on TLL's selected one.
  Outputs go to `results/e3d_unknown_structure/`.
- **E3e**: on E3d's truths and samples, the 2 x 2 of NPCC fitted simplified
  or non-simplified on TLL's selected or on the true structure. Outputs go to
  `results/e3e_simplification_2x2/` (d = 5, beside E3d's) and
  `results/e3e_simplification_2x2_gpu053/` (d = 10).
- **E10 on E3d**: on the same truths and samples, equal-weight mixtures of
  K = 5 vines, one per structure, for NPCC and for each competitor. Outputs
  go to `results/e10_e3d_2026-11-11/`, and TabPFN's at d = 10 on `decay` to
  `results/e10_e3d_gpu053_2026-11-11/`.
- **E11 on E3d**: on the same truths and samples, no vine at all: an
  equal-weight mixture of K = min(5, d!) chain-rule copula densities, one
  per variable order, the orders distinct and drawn uniformly at random,
  for NPCC. Outputs go to their own folder.

Two environments are involved. VDC's code runs only in its own (`~/vdc/env`,
Python 3.10, pyvinecopulib 0.7.5). Everything else runs in npcc's (`uv run
--no-sync python ...`, pyvinecopulib 1.0). Nothing from VDC is installed into
npcc's `.venv`. Where an npcc-side fit needs VDC's estimate — the adapted VDC
pair copula of E3a's cross-check and of E3c — it gets it from
`e3_vdc_server.py`, a process in VDC's environment.

## The runs the paper cites

The paper's E3a to E3d rows are the summaries `e3_collect.py` wrote in the
finish of the probit rerun (TAG 2026-11-10, b7702e1; `../README.md`), at
03:05 on 2026-11-11: `results/e3_vdc/e3{a,b,c}_summary.csv` and
`results/e3d_unknown_structure/e3d_{summary,paired}.csv`. The rerun refitted
every NPCC arm; every competitor row was reused from the runs before E8, and
the rerun checked that none of those files changed. Every row records the
commit it ran from in `npcc_commit`:

| rows | commit | how they ran |
|---|---|---|
| E3a, E3b: every NPCC arm | b7702e1 | the rerun: the gradient boosters in lane C, one core each, `--device cpu`; the in-context learners in lane B (`e3_run_gpu_arms.py --main-tabpfn v3 --stages e3a e3b --no-collect`) |
| E3a, E3b: VDC, TLL, parametric and the references | c2f0cb7 | reused |
| E3a: `vdc_adapted`, the cross-check | 349ec08 | reused |
| E3c: every NPCC arm and `rosenblatt-vinecop-check` | b7702e1 | the rerun: the boosters in lane C; TabPFN v3 in lane V, d = 10 and then d = 5 on each design |
| E3c `decay`: VDC, TLL, `tll_cpp` and parametric | d10182b | reused |
| E3c `strong`: the same | d18480b | reused |
| E3c: VDC's own vine (`vdc_own`), both designs | d10182b | reused |
| E3d: CatBoost's `random-ns`, `random-s`, `tll-ns` and `true-s` | b7702e1 | the rerun's lane C |
| E3d: TabPFN v3's `random-ns` and `tll-ns` | b7702e1 | the rerun's lane V: d = 5 on both designs; d = 10 on `decay`, ten seeds; `tll-ns` at d = 10 on `strong`, seeds 101-505 |
| E3d: VDC, TLL, parametric, `tll_true` and `parametric_true` | 2b319b7 | reused |
| E3d: `vdc_true` | 4b83733; 2b319b7 in one d = 10 seed of each design | reused |
| `e3_nan_diagnostics.csv` | 73a78c9 | not rerun: its rows are parametric fits, which the rerun reused |

E3e and E10 on E3d, which `tab:unknown-structure` adds, ran later from
d9dc80e; their sections below have their records.

**Logs.** The rerun's NPCC logs are in `results/rerun_2026-11-10/logs/`
(`laneC_*.log`, `laneV.log`, `e3ab_gpu.log`, `e3_collect.log`). Of the logs
in `results/e3_vdc/`, the `*_vdc_side.log`, `*_competitors.log`,
`e3a_npcc_vdc_adapted.log` and `table1_rerun_cpu.log` belong to reused rows;
`e3a_npcc_gpu.log` and `e3b_npcc_gpu.log` hold every run's in-context arms,
appended in turn, the rerun's probit lines last; the other `*_npcc_*.log`
files, `gpu_mem_monitor.log` and `gpu_queue.log` belong to superseded NPCC
arms. Likewise, of the `e3d_unknown_structure_*.log` files at the results
root, `truth`, `vdc_side` and `competitors*` belong to reused rows, and
`catboost_*` and `gpu_*` to superseded arms.

### Superseded

- The NPCC arms from before E8, on logit, written beside the competitor rows
  above on 2026-11-05 and 06. The logit rerun (TAG 2026-11-09) moved them,
  with the collector's tables of the time, into
  `results/superseded_2026-11-06/` (`MANIFEST.txt`).
- The logit rerun's own NPCC arms, from 50d2ca5, moved by the probit rerun
  into `results/superseded_2026-11-10/` (`MANIFEST.txt`).

## The scripts

| Script | Env | What it does |
|---|---|---|
| `e3_common.py` | both | Paths, provenance (versions, commits, the sha256 of every E3 module a process loaded), a byte-reproducible `.npz` writer, CSV/JSON writers |
| `e3a_vdc_side.py` | VDC | E3a samples at both `n_train` (the runner's generator), the true grid density, VDC / TLL / parametric fits with KL, IAE, ISE and the grid maxima, and the Table 1 reproduction check |
| `e3a_npcc_fit.py` | npcc | E3a NPCC arms, and `vdc-adapted`: VDC as a pyvinecopulib 1.0 pair copula, a cross-check of VDC's own lookup |
| `e3b_vdc_side.py` | VDC | E3b samples (the stress test's generator, rebuilt: VDC's is not released), the oracle and pooled oracle, VDC / TLL / parametric pooled, and per-regime references |
| `e3b_npcc_fit.py` | npcc | E3b NPCC arms, conditioning on `U1` and pooled |
| `e3c_vdc_side.py` | VDC | E3c, `strong` design: true vines and samples, and VDC's own vine on the true order (a reference) |
| `e3c_decay_vdc_side.py` | VDC | E3c, `decay` design: the same orders and edge cases, tau scaled by 0.7 per tree, and VDC's own vine |
| `e3c_npcc_fit.py` | npcc | E3c arms in one vine machinery: VDC, TLL and NPCC pairs in `e3_vine.SharedVinecop`; parametric selection and a TLL check in pyvinecopulib's compiled vine |
| `e3d_structure.py` | both | Vine edges from a structure matrix or a D-vine order, and their recovery of a true structure |
| `e3d_truth.py` | npcc | E3d truths: uniform random R-vines (E2's sampler), the nine cases, both designs, a random structure per cell |
| `e3d_vdc_side.py` | VDC | E3d `vdc`: VDC's own pipeline, choosing its own greedy D-vine |
| `e3d_npcc_fit.py` | npcc | E3d `parametric` and `tll` with Dissmann selection, true-structure references, and NPCC on random, TLL-selected and true structures; E3e's variants |
| `e3d_mixture.py` | npcc | E10 on E3d: mixtures of vines on TLL's selected structure and four weighted random ones, for NPCC, TLL, parametric and VDC's pairs |
| `e3d_chain.py` | npcc | E11 on E3d: mixtures of chain-rule copula densities along five random variable orders, for NPCC |
| `e3e_report.py` | npcc | E3e's 2 x 2: each variant's mean excess NLL and the paired effects of simplifying and of the structure |
| `e3_vdc_server.py` | VDC | VDC's density estimate as a service (started by `e3_vdc_pair.VdcServer`) |
| `e3_vdc_pair.py` | npcc | The server client and `VdcBicop`: VDC's 64 x 64 grid replicate-padded and held by pyvinecopulib 1.0's `TorchTllBicop` with `norm_maxiter=0`, adapted from pyvinecopulib PR #217 |
| `e3_vine.py` | npcc | `SharedVinecop`: a simplified pyvinecopulib 1.0 `VinecopBase` hosting any torch pair copula |
| `e3_npcc.py` | npcc | E1's estimators, at the probit transform (same labels and specs), and one timed fit |
| `e3_collect.py` | npcc | Long tables, summaries, cross-checks, the common-set scoring, and the code check |
| `e3_nan_diagnostics.py` | npcc | Which arms, trees and pair copulas give non-finite vine densities |
| `e3_run_gpu_arms.py` | npcc | The pending GPU arms, once the GPU is free |
| `e6_timing.sh` | both | E6's part for §3.6's `tab:cost`: clean timings of E3a's pairs and E3d's vines, one job at a time |
| `e6_timing.py` | npcc | Its vine timings (competitors with and without selection, the TabPFN v3 vine) and `e3_timing.csv` |
| `e6_timing_vdc.py` | VDC | Its VDC vine timings, with and without VDC's order selection |

Each script documents its design, metric and outputs at the top.

## VDC's environment (once)

VDC v0.2.0 is the tag commit `e09f3d3`. Its tree, `db4fe77`, is identical
to that of `d5aaa46`, the release's merge into `main`.

```bash
git clone https://github.com/KempnerInstitute/vine-denoising-copula ~/vdc/vine-denoising-copula
git -C ~/vdc/vine-denoising-copula checkout v0.2.0        # e09f3d3

mamba create -y -p ~/vdc/env python=3.10 pip               # a conda prefix env; 3.10.22 here
cd ~/vdc/vine-denoising-copula
~/vdc/env/bin/python -m pip install -r paper/neurips2026/runners/constraints.txt
~/vdc/env/bin/python -m pip install --no-deps -e .
```

The scripts find VDC through constants at the top of `e3_common.py`:
`VDC_REPO`, `VDC_PYTHON`, `CHECKPOINT` and `NPCC_REPO` (this machine's
`/home/thibault/...` paths). On another machine, set `E3_NPCC_REPO` (the npcc
checkout whose `results/` is read and written), `E3_NPCC_PYTHON` (its
interpreter, `<repo>/.venv/bin/python` by default), `E3_VDC_REPO` and
`E3_VDC_PYTHON`. Without VDC's interpreter, the npcc-only arms still run, and
their rows hold `None` for VDC's versions.

`constraints.txt` pins the numerical stack of VDC's paper replays: torch
2.9.1, NumPy 2.2.6, SciPy 1.15.3, pyvinecopulib 0.7.5, networkx 3.4.2,
matplotlib 3.10.6, hydra-core 1.3.2, tqdm 4.67.3, h5py 3.13.0, PyYAML
6.0.2, scikit-learn 1.7.2 and pytest 8.3.5. `pip freeze` in the env shows
each at exactly that version, and VDC as `-e git+...@e09f3d35...`. The
editable install is what VDC's runners expect from `--expected-module-root`.

## The weights

```bash
mkdir -p ~/vdc/weights
cd ~/vdc/vine-denoising-copula
~/vdc/env/bin/python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1 \
    --cache-dir ~/vdc/weights
sha256sum ~/vdc/weights/vdc-denoiser-m64-v1/vdc-denoiser-m64-v1.pt
# 8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f
```

The download comes from Hugging Face, `hsafaai/vdc-denoiser-m64-v1`
(532 MB). Every E3 script that loads the model refuses any other sha256.

## Commands, and the files they write

Every NPCC arm is one of E1's estimators at the probit transform, the only
one the paper presents (E8, in `../README.md`); `e3_npcc.py` lists them. The
runs before E8 used the same commands with E1's logit estimators. "The runs
the paper cites" (above) says which run wrote each row.

Run everything from the npcc repository root. The CPU runs pin each process
to one core and one BLAS/OpenMP thread, as VDC's runner times its methods;
those are the settings below, with `nice -n 10` because the machine was
shared. The core numbers are what the run before E8 used and only matter for
timings. First, in the shell that runs them:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

### VDC's Table 1, reproduced with its own runner

```bash
cd ~/vdc/vine-denoising-copula
nice -n 10 ~/vdc/env/bin/python paper/neurips2026/runners/reproduce_table1_likelihood.py \
    --checkpoint ~/vdc/weights/vdc-denoiser-m64-v1/vdc-denoiser-m64-v1.pt \
    --expected-module-root ~/vdc/vine-denoising-copula \
    --device cpu --output ~/npcc/results/e3_vdc/table1_rerun_cpu \
    > ~/npcc/results/e3_vdc/table1_rerun_cpu.log 2>&1
```

This writes `table1_rerun_cpu/results.json` and `likelihoods.npz`, with the
runner unmodified. The means are VDC 0.0061112, TLL (its BIC choice between
independence and TLL) 0.0134801 and parametric 0.00036485; the published
ones are 0.00611124, 0.01348007 and 0.00036485.

### E3a

```bash
taskset -c 12 nice -n 10 ~/vdc/env/bin/python scripts/paper/e3/e3a_vdc_side.py \
    --reference-run results/e3_vdc/table1_rerun_cpu --out results/e3_vdc/e3a \
    > results/e3_vdc/e3a_vdc_side.log 2>&1
taskset -c 12 nice -n 10 uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
    --estimators lightgbm-probit --n-train 2000 10000 --device cpu \
    > results/e3_vdc/e3a_npcc_lightgbm.log 2>&1
for est in catboost xgboost; do
  taskset -c 15 nice -n 10 uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
      --estimators $est-probit --n-train 2000 10000 --device cpu \
      > results/e3_vdc/e3a_npcc_$est.log 2>&1
done
taskset -c 12 nice -n 10 uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
    --estimators vdc-adapted --vdc-core 12 --n-train 2000 10000 --device cpu \
    > results/e3_vdc/e3a_npcc_vdc_adapted.log 2>&1
```

`e3a/` then holds:
- `samples.npz` and `samples_n10000.npz`;
- `grid_truth.npz`, the true density on the 200 x 200 midpoint grid;
- `vdc_grids_n{2000,10000}.npz`, VDC's 64 x 64 grids;
- `vdc_side.csv` and `vdc_side.json`, the VDC-env rows and the reproduction
  check;
- `npcc_cells/<estimator>/n<n_train>/`, one JSON row and one `.npy` of
  test-point log-densities per fit;
- `npcc_<estimator>.csv`, including `npcc_vdc-adapted.csv`, the cross-check.

The independence probe — the largest fitted density on the grid and where it
occurs — is part of every row (`grid_max_pdf`, `grid_argmax_u`,
`grid_argmax_v`). The collector tabulates it.

### E3b

```bash
taskset -c 13 nice -n 10 ~/vdc/env/bin/python scripts/paper/e3/e3b_vdc_side.py \
    --out results/e3_vdc/e3b > results/e3_vdc/e3b_vdc_side.log 2>&1
for est in lightgbm xgboost; do
  taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3b_npcc_fit.py \
      --estimators $est-probit --device cpu > results/e3_vdc/e3b_npcc_$est.log 2>&1
done
taskset -c 15 nice -n 10 uv run --no-sync python scripts/paper/e3/e3b_npcc_fit.py \
    --estimators catboost-probit --device cpu > results/e3_vdc/e3b_npcc_catboost.log 2>&1
```

`e3b/` then holds `samples.npz`, `vdc_side.csv`, `vdc_side.json`,
`npcc_cells/<estimator>/` and `npcc_<estimator>.csv`.

### E3c

The two designs share `e3c_npcc_fit.py`, which takes the design's folder.
For the primary `decay` design (`results/e3_vdc/e3c_decay/`; the run before
E8 used cores 10–13):

```bash
taskset -c 10 nice -n 10 ~/vdc/env/bin/python scripts/paper/e3/e3c_decay_vdc_side.py \
    --out results/e3_vdc/e3c_decay > results/e3_vdc/e3c_decay_vdc_side.log 2>&1
F=results/e3_vdc/e3c_decay
taskset -c 10 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py --folder $F \
    --arms tll tll_cpp parametric vdc --vdc-core 10 > results/e3_vdc/e3c_decay_npcc_competitors.log 2>&1
taskset -c 11 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py --folder $F \
    --arms npcc-lightgbm-probit > results/e3_vdc/e3c_decay_npcc_lightgbm.log 2>&1
taskset -c 12 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py --folder $F \
    --arms npcc-xgboost-probit > results/e3_vdc/e3c_decay_npcc_xgboost.log 2>&1
taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py --folder $F \
    --arms npcc-catboost-probit rosenblatt-vinecop-check > results/e3_vdc/e3c_decay_npcc_catboost.log 2>&1
```

For the `strong` stress variant (`results/e3_vdc/e3c/`, the default folder):

```bash
taskset -c 11 nice -n 10 ~/vdc/env/bin/python scripts/paper/e3/e3c_vdc_side.py \
    --out results/e3_vdc/e3c > results/e3_vdc/e3c_vdc_side.log 2>&1
taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py \
    --arms tll tll_cpp parametric vdc --vdc-core 13 > results/e3_vdc/e3c_npcc_competitors.log 2>&1
taskset -c 11 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py \
    --arms npcc-lightgbm-probit > results/e3_vdc/e3c_npcc_lightgbm.log 2>&1
taskset -c 14 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py \
    --arms npcc-xgboost-probit > results/e3_vdc/e3c_npcc_xgboost.log 2>&1
taskset -c 15 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py \
    --arms npcc-catboost-probit rosenblatt-vinecop-check > results/e3_vdc/e3c_npcc_catboost.log 2>&1
```

Each design's folder then holds:
- `samples.npz`: per cell, the train and test samples, true log-density,
  order, structure matrix and edge cases, plus each edge's tau for `decay`;
- `vdc_side.csv` (VDC's own vine) and `vdc_side.json`, whose `design` key is
  `"decay"` for the decay design (the strong design's predates the key);
- `npcc_cells/<arm>/` and `npcc_<arm>.csv`.

The collector labels every merged E3c row with its `design`.

The run before E8, whose VDC-side and competitor rows the paper cites, ran
E3a, E3b and the strong design as five chains, one per core (11–15),
starting each npcc-side command once its VDC side had written
`vdc_side.json`. An npcc-side script needs nothing else, so the order
across sub-experiments is free.

### E3d (unknown structure)

```bash
taskset -c 10 nice -n 10 uv run --no-sync python scripts/paper/e3/e3d_truth.py \
    > results/e3d_unknown_structure_truth.log 2>&1
taskset -c 10 nice -n 10 ~/vdc/env/bin/python scripts/paper/e3/e3d_vdc_side.py \
    > results/e3d_unknown_structure_vdc_side.log 2>&1
E=scripts/paper/e3/e3d_npcc_fit.py
taskset -c 11 nice -n 10 uv run --no-sync python $E \
    --arms parametric tll parametric_true tll_true vdc_true --vdc-core 11
for v in random-ns tll-ns random-s true-s; do
  taskset -c 12 nice -n 10 uv run --no-sync python $E --arms npcc-catboost-probit-$v
done
# GPU, main-text TabPFN (v3), arms (a) and (b) only (the paper lead's scope):
# d = 5 on both designs, d = 10 on the decay design and five seeds.
for v in random-ns tll-ns; do
  uv run --no-sync python $E --arms npcc-tabpfn-quantiles-v3-probit-$v --dims 5 \
      --device cuda --gpu-is-free >> results/e3d_unknown_structure_gpu.log 2>&1
  uv run --no-sync python $E --arms npcc-tabpfn-quantiles-v3-probit-$v --designs decay --dims 10 \
      --seeds 101 202 303 404 505 --device cuda --gpu-is-free >> results/e3d_unknown_structure_gpu.log 2>&1
done
```

The truths are random R-vines drawn uniformly with pyvinecopulib 1.0's
`RVineStructure.sample` (E2's sampler), for d ∈ {5, 10} and seeds 101–1010,
in both designs (paired: same structure and edge cases). The samples are
drawn with pyvinecopulib 1.0. `results/e3d_unknown_structure/` then holds
`samples.npz`, `truths.json`, `vdc_side.csv` and `vdc_side.json`,
`cells/<arm>/` and `npcc_<arm>.csv`. The collector adds `e3d_long.csv`,
`e3d_summary.csv` and `e3d_paired.csv` there.

The competitors' `fit_s` includes their structure selection; arm (b)'s
includes TLL's Dissmann selection, which it reuses. Every row records the
structure it used and its recovery of the true one: the share of true edges
found in each tree and overall, and whether it is the truth.

### E3e (simplified or not, on a selected or the true structure)

E3e asks how much of NPCC's E3d result is the non-simplified fit and how much
the structure. It crosses the two:

| | TLL's selected structure | true structure |
|---|---|---|
| non-simplified | `tll-ns` (E3d's arm (b)) | `true-ns` |
| simplified | `tll-s` | `true-s` |

`e3d_npcc_fit.py` reads E3d's `samples.npz` and `truths.json` from `--folder`
(E3d's, by default) and writes its cells under `--out`. `tll-s` and
`true-ns` refuse E3d's folder, which the paper cites. Each variant's torch
seed is `derive_seed("E3d", design, d, seed, variant, estimator_id)`, as
E3d's.

Where each part runs:
- **d = 5, both designs, here.** E3d holds TabPFN v3's `tll-ns` at d = 5,
  ten seeds on each design, and this machine reproduces it bit for bit
  (E10's member 1 below is that fit), so E3e runs the other three variants.
- **d = 10, on gpu053, all four variants**, one design at a time, as a whole
  experiment of that machine. The same d = 10 fit there and here differs by
  up to about 4e-3 in excess NLL, the size of some of the effects, so a 2 x 2
  takes no cell from the other machine. E3d's own d = 10 `tll-ns` (ten
  seeds on `decay`, five on `strong`) is not used.

```bash
E=scripts/paper/e3/e3d_npcc_fit.py
# Here, TabPFN v3: d = 5 on both designs (20 cells per variant).
F=results/e3e_simplification_2x2
for v in tll-s true-ns true-s; do
  uv run --no-sync python $E --folder results/e3d_unknown_structure \
      --arms npcc-tabpfn-quantiles-v3-probit-$v --out $F --dims 5 --designs decay strong \
      --device cuda --gpu-is-free >> $F/logs/tabpfn_d5_$v.log 2>&1
done
# On gpu053, TabPFN v3: d = 10, one design (D) at a time, in two batches of
# five seeds, every variant; then copied back as results/e3e_simplification_2x2_gpu053.
F=results/e3e_simplification_2x2
for seeds in "101 202 303 404 505" "606 707 808 909 1010"; do
  for v in tll-ns tll-s true-ns true-s; do
    uv run --no-sync python $E --folder results/e3d_unknown_structure \
        --arms npcc-tabpfn-quantiles-v3-probit-$v --out $F --designs $D --dims 10 \
        --seeds $seeds --device cuda --gpu-is-free >> $F/logs/tabpfn_d10_${D}_$v.log 2>&1
  done
done
```

Every run resumes by cell: a cell whose JSON row exists is skipped.
`$F/cells/<arm>/` and `$F/npcc_<arm>.csv` then hold the variants. At d = 5
the fourth is `results/e3d_unknown_structure/cells/npcc-tabpfn-quantiles-v3-probit-tll-ns/`.
`e3_collect.py` reads neither folder for E3e.

`e3e_report.py` reads the variants' CSVs and prints, per design and d, the
four cells' mean excess NLL and four paired effects: simplifying on each
structure, and TLL's structure against the true one under each fit. With
`--out` it writes `e3e_wide.csv` and `e3e_effects.csv`. Give it folders from
one machine; where E3d's folder supplies `tll-ns`, it goes first:

```bash
R=scripts/paper/e3/e3e_report.py
uv run --no-sync python $R results/e3d_unknown_structure results/e3e_simplification_2x2 \
    --dims 5 --out results/e3e_simplification_2x2/analysis_d5
# The folder held the decay design only when this ran, hence the name.
uv run --no-sync python $R results/e3e_simplification_2x2_gpu053 \
    --dims 10 --out results/e3e_simplification_2x2_gpu053/analysis_d10_decay
```

**Ran.**
- d = 5, here, from a clean worktree at d9dc80e (the tip of E10's pull
  request, #51, before the base was merged into it), one GPU process at a
  time on 2026-11-11: `tll-s` 21:54 to 22:15, `true-ns` to 22:38, `true-s`
  to 22:59. `e3e_report.py` wrote `analysis_d5/`.
- d = 10 `decay`, on gpu053 from a checkout at d9dc80e, on 2026-11-12:
  seeds 101-505 from 00:23 to 02:07 and seeds 606-1010 from 03:59 to 05:44,
  about 26 min per variant and batch, interleaved with E10's batches below.
  Copied back unchanged as `results/e3e_simplification_2x2_gpu053`, whose
  `env_gpu053.txt` records the machine and whose `logs/` holds the queue
  and its log. `e3e_report.py` on `decay` alone wrote `analysis_d10_decay/`
  there. Its `tll-ns` differs from E3d's, fitted here, by -0.0026 to
  +0.0042 per seed, -4e-5 on average.
- d = 10 `strong`, on gpu053 under the same checkout, is queued behind
  E10's conditional run there (`gpu053_queue_v3.sh`).

**Check (CPU, CatBoost, `decay`, d = 5, seed 101).** Run into a scratch
folder, `tll-ns` and `true-s` reproduced E3d's CatBoost cells bit for bit
(the `.npy` files are equal). `true-ns` fitted the true matrix and `tll-s`
TLL's selected one (E3d's `tll` row), each in the context its name says. The
excess NLLs were `tll-ns` 0.2350, `tll-s` 0.1535, `true-ns` 0.2175 and
`true-s` 0.1146.

### E10 on E3d (mixtures over structures)

E2's `mixture` arm (`configs/vine_study_mixture*.toml`) asks the question in
the conditional vine study; `e3d_mixture.py` asks it on E3d's unconditional
truths, where the competitors can answer too. Per cell, K = 5 structures are
picked on the training data:
- member 1 is TLL's Dissmann selection (`vine_runner.select_structure`),
  E3d's `tll` and `tll-ns` structure;
- members 2 to 5 are pyvinecopulib's weighted random spanning trees, with
  weights |tau| (`vine_runner.weighted_random_structure`), seeded
  `derive_seed("E3d", "mixture", d, seed, j)`.

Every arm fits one vine per structure and scores the members' equal-weight
mixture. The arms are:
- `tll` and `parametric`, in pyvinecopulib's compiled vine with E3d's
  controls and the structure given;
- `vdc_shared`, VDC's pair copulas in `e3_vine.SharedVinecop`, as E3d's
  `vdc_true`;
- `npcc-<estimator>`, non-simplified. Member 1's torch seed is E3d's for
  `tll-ns`, so NPCC's member 1 is E3d's arm (b).

Each row records:
- the mixture's excess NLL;
- each member's, on its own finite points and on the mixture's;
- the members' mean, member 1's and the best member's;
- every structure, its recovery of the truth, and how many of the K
  structures are distinct vines, compared as edge sets;
- the seconds of every member.

`summary.csv` has one row per cell and method with the comparisons, all on
the mixture's finite points. A cell refuses to run if another arm's row for
it records other structures, and the summary refuses methods of one cell
that mixed different ones. Members are fitted, evaluated and released one at
a time, and saved as they finish, so a cell resumes at its next member.

```bash
M=scripts/paper/e3/e3d_mixture.py
O=results/e10_e3d_2026-11-11
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# CPU, the competitors: d = 5 on both designs, d = 10 on decay.
taskset -c 4 nice -n 10 uv run --no-sync python $M --arms tll parametric --designs decay strong \
    --dims 5 --out $O > $O/logs/tll_parametric_d5.log 2>&1
taskset -c 5 nice -n 10 uv run --no-sync python $M --arms tll parametric --designs decay \
    --dims 10 --out $O > $O/logs/tll_parametric_d10_decay.log 2>&1
taskset -c 6 nice -n 10 uv run --no-sync python $M --arms vdc_shared --vdc-core 7 --designs decay strong \
    --dims 5 --out $O > $O/logs/vdc_shared_d5.log 2>&1
taskset -c 6 nice -n 10 uv run --no-sync python $M --arms vdc_shared --vdc-core 7 --designs decay \
    --dims 10 --out $O > $O/logs/vdc_shared_d10_decay.log 2>&1
# CPU, the competitors and CatBoost at d = 10 on strong (with --folder, which
# a worktree outside the main checkout needs).
E3D=results/e3d_unknown_structure
taskset -c 4 nice -n 10 uv run --no-sync python $M --folder $E3D --arms tll parametric \
    --designs strong --dims 10 --out $O > $O/logs/tll_parametric_d10_strong.log 2>&1
taskset -c 5 nice -n 10 uv run --no-sync python $M --folder $E3D --arms npcc-catboost-probit \
    --designs strong --dims 10 --out $O > $O/logs/catboost_d10_strong.log 2>&1
taskset -c 6 nice -n 10 uv run --no-sync python $M --folder $E3D --arms vdc_shared --vdc-core 7 \
    --designs strong --dims 10 --out $O > $O/logs/vdc_shared_d10_strong.log 2>&1
# GPU, TabPFN v3, here: d = 5 on both designs, then d = 10 on strong, one
# process per seed.
uv run --no-sync python $M --arms npcc-tabpfn-quantiles-v3-probit --designs decay strong --dims 5 \
    --out $O --device cuda --gpu-is-free >> $O/logs/tabpfn_d5.log 2>&1
for s in 101 202 303 404 505 606 707 808 909 1010; do
  uv run --no-sync python $M --folder $E3D --arms npcc-tabpfn-quantiles-v3-probit --designs strong \
      --dims 10 --seeds $s --out $O --device cuda --gpu-is-free >> $O/logs/tabpfn_d10_strong.log 2>&1
done
# GPU, TabPFN v3, on gpu053: d = 10 on decay, in two batches of five seeds,
# into a copy of $O (so its structures are checked against the competitors').
for seeds in "101 202 303 404 505" "606 707 808 909 1010"; do
  uv run --no-sync python $M --arms npcc-tabpfn-quantiles-v3-probit --designs decay --dims 10 \
      --seeds $seeds --out $O --device cuda --gpu-is-free >> $O/logs/tabpfn_d10_decay.log 2>&1
done
```

On gpu053, `E3_NPCC_REPO` names the checkout there, whose
`results/e3d_unknown_structure/` holds a copy of E3d's `samples.npz` and
`truths.json` (or pass `--folder`). That copy of `$O` comes back whole as
`results/e10_e3d_gpu053_2026-11-11`, which keeps gpu053's TabPFN cells apart
from this machine's: the same fit on the two machines differs by about 1e-3
at d = 10 (E3e above). Its `summary.csv` already has every method at d = 10
on `decay`.

`scripts/paper/e10/compare_e3d.py` scores the TabPFN mixture against every
reference on each cell's common finite points (its docstring lists them),
and writes `compare_cells.csv` and `compare.csv`:

```bash
C=scripts/paper/e10/compare_e3d.py
uv run --no-sync python $C --e10 results/e10_e3d_2026-11-11 --dims 5 \
    --out results/e10_e3d_2026-11-11/analysis_d5
uv run --no-sync python $C --e10 results/e10_e3d_gpu053_2026-11-11 --designs decay --dims 10 \
    --out results/e10_e3d_gpu053_2026-11-11/analysis_d10
uv run --no-sync python $C --e10 results/e10_e3d_2026-11-11 --designs strong --dims 10 \
    --out results/e10_e3d_2026-11-11/analysis_d10_strong
```

`$O` holds:
- `cells/<arm>/<design>_d<d>_s<seed>.json`, the row;
- `.npy` beside it, the mixture's test-point log-densities, and
  `_members.npy`, the members' (K x 20,000), both float32 with NaN kept;
- `cells/<arm>/members/`, each finished member's row and float64
  log-densities;
- `mixture_<arm>.csv` (the rows) and `summary.csv`. The summary also gives,
  for `tll` and NPCC, the largest difference between member 1's
  log-densities and those of E3d's `tll` or `tll-ns` on the same cell (0
  when member 1 reproduces it bit for bit in float32).

**Ran (CPU).** The competitors, and NPCC with CatBoost as a CPU preview of
the TabPFN arm, ran on 2026-11-11 from a worktree of `e10-e3d` (98c9022 to
1d9816a; no commit between them changed what a cell computes), one thread
per process, into `results/e10_e3d_competitors_2026-11-11`. That folder was
renamed `results/e10_e3d_2026-11-11` before the last runs, so that the
TabPFN arm writes beside the competitors. Each run's launcher and log are in
its `logs/`:
- `run.sh`: `tll` and `parametric` at d = 5 on both designs and at d = 10 on
  `decay`, and `vdc_shared` at d = 5;
- `run_vdc_d10.sh` and `run_catboost.sh`: VDC's pairs at d = 10 and
  CatBoost, stopped part-way when the cores were needed elsewhere;
- `run_rest_cores10-11.sh`: the rest of both, resumed by cell and member.

Every row's `npcc_commit` is the shared checkout's, as `e3_common`
records it; `script_repo_commit` is the worktree's.

**Checks.**
- Every cell's arms mixed the same five structures, and member 1's is E3d's
  `tll` structure.
- CatBoost's member 1 reproduced E3d's `npcc-catboost-probit-tll-ns` bit for
  bit on every cell.
- `tll`'s member 1 reproduced E3d's `tll` bit for bit on 29 of its 30 cells.
  On `decay_d10_s1010` its excess NLL is 1.2e-4 above E3d's. The two fits
  share the structure, but pyvinecopulib's fit on a given structure and its
  Dissmann pass give slightly different TLL pairs in trees 6 to 9 there
  (densities about 2e-3 apart), so this is pyvinecopulib's and not the
  mixture's.
- With `--k 1`, the mixture is E3d's `tll`. Member 1, mixed with K - 1 copies
  of itself, scores as itself to within 4e-15.
- On every cell, the mixture's excess NLL is at most the members' mean.

**Ran (TabPFN v3).**
- d = 5, both designs, here, from a clean worktree at d9dc80e (the tip of
  E10's pull request, #51, before the base was merged into it), one GPU
  process, 20:03 to 21:54 on 2026-11-11, into `$O`. Its member 1 reproduces
  E3d's `npcc-tabpfn-quantiles-v3-probit-tll-ns` bit for bit on every cell.
  `compare_e3d.py --dims 5` wrote `$O/analysis_d5/`.
- d = 10 `decay`, on gpu053 from a checkout at d9dc80e, into its copy of
  `$O`: seeds 101-505 from 22:22 on 2026-11-11 to 00:23, and seeds
  606-1010 from 02:07 to 03:59 on 2026-11-12. Its member 1 is gpu053's E3e
  `tll-ns` bit for bit. Copied back as `results/e10_e3d_gpu053_2026-11-11`,
  with `env_gpu053.txt` and the queue in `logs/`; `compare_e3d.py` wrote its
  `analysis_d10/` here.
- d = 10 `strong`, here, from the same worktree at d9dc80e, on 2026-11-12:
  TabPFN one seed at a time on the GPU, 11:22 to 15:17 (`local_queue4.sh`
  in `$O/logs/`, about 26 minutes a seed), and the competitors and CatBoost
  on cores 4 to 7, 10:58 to 11:40 (`cpu_d10_strong_v2.sh`). A first CPU
  launcher left out `--folder` and stopped at start with nothing written;
  its logs are the `*_v1_failed_no_folder.log` files. Member 1 reproduces
  E3d's `npcc-tabpfn-quantiles-v3-probit-tll-ns` bit for bit on seeds 101
  to 505, the five E3d holds at d = 10 `strong`. `compare_e3d.py --designs
  strong --dims 10`, from this branch at aa48017, wrote
  `$O/analysis_d10_strong/` and its `provenance.txt`.

### E11 on E3d (chain-rule densities, no vine)

E11 asks why a vine is needed at all (`scripts/paper/README.md`, E11).
`e3d_chain.py` writes the copula density down by the chain rule along a
variable order, `c(u) = prod_{k=2..d} f(u_pi(k) | u_pi(<k))`. Each factor
is one Rosenblatt direction of the estimator
(`npcc.experiments.chain.ChainCopula`), here conditioning on the earlier
scores alone, since E3d has no `x`. An order costs `d - 1` regressions:
- per cell, K = min(5, d!) distinct orders drawn uniformly at random,
  order k from a generator seeded by
  `derive_seed("E3d", "chain", design, d, seed, k)`, which draws again where
  its order repeats an earlier one (`chain.draw_orders`). The d = 5 run from
  fa7aee3 kept a repeat in one cell (`decay`, seed 202), and is superseded
  by a rerun under this rule; at d = 10 every cell's orders were already
  distinct, so that run's are this rule's;
- factor f of order k has the torch seed
  `derive_seed("E3d", "chain", design, d, seed, k, f, estimator_id)`;
- `--estimator` is an E1 label, read from E1's config as the other E3
  scripts read it (`tabpfn-quantiles-v3-probit` by default);
- the mixture and every order are scored as E10 on E3d scores its members
  (`e3d_mixture.score_members`), on the 20,000 test points.

Orders are fitted one at a time, each evaluated and released before the
next, and saved as they finish, so a cell resumes at its next order. The
CLI is `e3d_mixture.py`'s, with `--estimator` for `--arms` and no
`--vdc-core`.

```bash
C=scripts/paper/e3/e3d_chain.py
O=results/e11_e3d_<TAG>
mkdir -p $O/logs
# GPU, TabPFN v3, here: d = 5 on both designs.
uv run --no-sync python $C --designs decay strong --dims 5 --out $O \
    --device cuda --gpu-is-free >> $O/logs/tabpfn_d5.log 2>&1
# GPU, TabPFN v3, on gpu053: d = 10, decay then strong, in batches of five
# seeds, into its own folder there.
for design in decay strong; do
  for seeds in "101 202 303 404 505" "606 707 808 909 1010"; do
    uv run --no-sync python $C --designs $design --dims 10 --seeds $seeds --out $O \
        --device cuda --gpu-is-free >> $O/logs/tabpfn_d10_$design.log 2>&1
  done
done
```

As for E10 on E3d, on gpu053 `E3_NPCC_REPO` names the checkout there, whose
`results/e3d_unknown_structure/` holds E3d's `samples.npz` and
`truths.json` (or pass `--folder`).

`$O` holds:
- `cells/chain-<estimator>/<design>_d<d>_s<seed>.json`, the row: the
  orders, their seeds and every factor's, the mixture's excess NLL and each
  order's (the `member` columns), the orders' mean and the best order on the
  mixture's finite points, and the seconds of every order;
- `.npy` beside it, the mixture's test-point log-densities, and
  `_members.npy`, the orders' (K x 20,000), both float32 with NaN kept;
- `cells/chain-<estimator>/members/`, each finished order's row and float64
  log-densities;
- `chain_<estimator>.csv`, the rows, and `chain_<estimator>_summary.csv`,
  per design and d over seeds.

`scripts/paper/e11/compare_e3d_chain.py` reads the chain against the vines,
paired by cell on the test points both have finite, with
`compare_e3d.py`'s pairing and columns. The references are:
- the chain's orders' mean and its best order;
- E3d's `npcc-<estimator>-tll-ns` and `vdc_true` (`--e3d`, E3d's folder by
  default);
- the same estimator's mixture over five structures in an E10-on-E3d
  folder, and its member 1, which refits `tll-ns` there (`--e10`);
- E3e's `tll-s`, `true-ns` and `true-s`, and `tll-ns` where that folder
  holds it (`--e3e`).

Each is named by the folder it came from. At d = 10, two machines' TabPFN
vines differ by up to 4e-3 in excess NLL per seed, so pass the E10 and E3e
folders fitted on the chain's machine and read their references:

```bash
uv run --no-sync python scripts/paper/e11/compare_e3d_chain.py --chain $O \
    --e10 results/e10_e3d_2026-11-11 --e3e results/e3e_simplification_2x2 \
    --dims 5 --out $O/analysis
```

It writes `compare_cells.csv` and `compare.csv`: per design, d and
reference, the mean difference chain - reference over seeds, its SE, and in
how many seeds the chain is lower.

**Ran.**
- d = 5, both designs, here, from the clean detached worktree at fa7aee3
  (#54's tip), one GPU process per seed, 15:45 to 16:12 on 2026-11-12,
  into `results/e11_e3d_2026-11-12` (`local_queue7.sh` in its `logs/`, with
  `COMMIT` and `env.txt`). `compare_e3d_chain.py` against E3d, E10 on E3d
  (`results/e10_e3d_2026-11-11`) and E3e (`results/e3e_simplification_2x2`),
  all made here, wrote `analysis_d5/`.
- d = 10, both designs, on gpu053 from a clean worktree at fa7aee3, into
  `e11_e3d_d10_2026-11-12` there: `decay` by its queue's version 4, 17:49
  to 18:25 on 2026-11-12, and `strong` by version 5, 18:30 to 19:01, in
  batches of five seeds (both queue scripts in its `logs/`). Copied back
  with every file's SHA-256 unchanged (`logs/COPIED_FROM.txt`). Every cell
  has five distinct orders. `compare_e3d_chain.py` wrote, here:
  - `analysis_d10_decay/`, against E3d (here), and E10 on E3d
    (`e10_e3d_gpu053_2026-11-11`) and E3e (`e3e_simplification_2x2_gpu053`)
    from gpu053;
  - `analysis_d10_strong/`, against E3d and E10 on E3d
    (`e10_e3d_2026-11-11`), both from this machine, so the chain's
    comparisons there cross machines (up to about 4e-3 per seed). E3e had
    no `strong` d = 10 cell yet, so `--e3e` held none.

  Each folder's `provenance.txt` has the command and says which machine made
  each reference.

**Cost, per cell, on one machine.** Fit plus scoring of the 20,000 test
points, from the rows' `fit_s` and `inference_s`, at d = 5, here: the chain
72.5 s on `decay` and 66.2 s on `strong`, E10's TabPFN mixture 330.3 s and
327.6 s, ratios of 0.22 and 0.20. At d = 10 `decay`, on gpu053: the chain
135.0 s and the mixture 1270.7 s, a ratio of 0.11. The chain took 128.1 s
at `strong` there, but that mixture ran on this machine, so there is no
ratio on one machine.

### The summaries

```bash
uv run --no-sync python scripts/paper/e3/e3_collect.py
```

This writes, in `results/e3_vdc/`:
- `e3a_long.csv`, `e3a_summary.csv` (per `n_train` and arm: KL, IAE and
  ISE with stratified standard errors, the paired KL difference to VDC,
  timings), `e3a_by_case.csv`, `e3a_table1_reproduction.csv`,
  `e3a_independence_probe.csv` and `e3a_vdc_crosscheck.csv`;
- `e3b_long.csv` and `e3b_summary.csv`;
- `e3c_long.csv`, `e3c_summary.csv` and `e3c_crosscheck.csv`, with a
  `design` column (`decay` rows first);
- `e3_code_check.csv` and `e3_summary.json`.

In `results/e3d_unknown_structure/` it writes `e3d_long.csv`,
`e3d_summary.csv`, and `e3d_paired.csv`, which holds the numbers to cite. Per
design and d, `e3d_paired.csv` gives:
- each NPCC arm's paired difference to VDC, parametric and TLL, with its
  standard error and the seeds it wins;
- (a) minus (b), and (c) minus (a), per backend;
- (b) minus VDC on the true structure;
- every arm's and competitor's structure recovery;
- the excluded test points, in total and per seed.

E3c and E3d are read from the per-cell JSON files rather than the per-arm
CSVs. A fit script writes its CSV only when its run ends, so a stage that is
running, or was stopped, appears up to its last finished cell, with
`n_seeds` stated.

### The GPU arms

The GPU arms run once the GPU is free, naming the main-text TabPFN version,
whose estimator is `tabpfn-quantiles-<version>-probit`:
- E3a at `n_train` = 2,000 runs every in-context learner (TabPFN v2.5, v3
  and v3.5, TabICL, Nori);
- E3a at `n_train` = 10,000 runs the main TabPFN version only;
- E3b runs every in-context learner;
- E3c, both designs, runs the main TabPFN version only (TabICL and Nori
  were dropped there for cost).

```bash
uv run --no-sync python scripts/paper/e3/e3_run_gpu_arms.py --main-tabpfn <v2.5|v3|v3.5>
```

It runs E3a, E3b and E3c's GPU arms, each in its own process and each
resuming from its per-fit files, then the collector. It removes
`PYTORCH_CUDA_ALLOC_CONF` from the children's environment, since
`expandable_segments` fails on this machine's vGPU.

### E6's timings (§3.6)

E6's driver, `scripts/paper/e6/run_all.sh`, runs this part last, on an idle
machine, one job at a time:

```bash
bash scripts/paper/e3/e6_timing.sh results/e6_timing
```

Everything goes under `results/e6_timing/e3/`, and the last step writes
`results/e6_timing/e3_timing.csv`, which E6's `consolidate.py` reads. Its
columns are section, method, setting, d, n, seed, stage, seconds, device,
compute, CPU model, threads and npcc commit, plus the case, the load average
and `loglik_gap_vs_e3`.
- **E3a**, `n_train` = 2,000: every method on the same nine samples (seed
  202 of each case), through E3a's own scripts and warm-ups:
  - VDC, TLL and parametric;
  - NPCC with TabPFN v3, TabICL, Nori, LightGBM, XGBoost and CatBoost.
- **E3d's vines**, on E3d's decay-design samples:
  - TabPFN v3's arm (a), at d = 5 (three seeds) and d = 10 (one seed);
  - TLL, parametric and VDC, each fit with and without structure selection,
    at d = 5 and d = 10 (three seeds). Without selection, the structure (or
    VDC's order) that the selection returned is given back, so the two fits
    differ by the selection alone.
- **Protocol:**
  - CPU methods run on one core, with every BLAS and OpenMP pool and torch
    at one thread;
  - GPU methods run with two host cores (`E6_GPU_THREADS`);
  - every timing follows an untimed warm-up in its process.
- **Check:** each timed fit records its held-out log-likelihood (KL in
  E3a), and `loglik_gap_vs_e3` is its distance to the row E3 reported for
  the same cell and method.
- **Wall time:** about 55 minutes, of which about 25 are the d = 10 TabPFN
  vine.
- **Variables:** `E6_CPU_CORE`, `E6_GPU_THREADS`, `E6_GPU_CORES`,
  `E6_E3A_SEEDS`, `E6_VINE_SEEDS` and `E6_TABPFN_D10_SEEDS` change the
  cores and the seeds.

## Reproducibility

- **Code.** Every row records, in `code_sha256`, the sha256 of the script
  that wrote it and of every E3 module that process had loaded. Rows that
  used the VDC server record the server's map too, in
  `vdc_server_code_sha256`. `e3_collect.py` looks each hash up in the
  module's git history and on disk, and checks the Table 1 runner's recorded
  `source_sha256` against the runner. It writes `e3_code_check.csv`, whose
  `version` column names the commit holding each version that ran. It exits
  with an error if any row came from code that is neither committed nor on
  disk (`--allow-stale` overrides).
- **Which commit each hash belongs to.** Some rows ran an earlier commit's
  version of a module, while the module has since changed in comments only.
  For example, rows produced before the American-spelling fix ran
  `e3_vdc_pair.py` and `e3a_vdc_side.py` as committed in `7000ae3`, and
  rows produced before "cell-centred" became "cell-centered" in the
  docstrings of `e3_vdc_pair.py` and `e3_vdc_server.py` ran those two as
  committed before that fix. Those rows stay valid and regenerable from
  their commits, and
  `e3_code_check.csv` maps every recorded hash to its commit. Each row also records
  the npcc commit, whether `src/npcc/core` was modified (`npcc_core_dirty`),
  and the modified paths elsewhere. These describe the checkout the scripts
  ran from (`e3_common.CODE_REPO`): a worktree's, when run from one with
  `PYTHONPATH` on its `src`. `e3_npcc.py` refuses a package imported from
  any other checkout.
- **Data.** Samples are drawn with pyvinecopulib 0.7.5 from fixed seeds, as
  each VDC-side script documents. The `.npz` files are written byte for byte
  reproducibly, so their sha256 recurs on regeneration. Every npcc-side
  script checks the sha256 recorded by its VDC side before fitting.
- **Determinism.** The VDC, TLL, parametric and CPU-backend fits are
  deterministic on one thread. Two runs of E3a's CPU arms agreed fit by fit
  (135 fits, identical KL). The GPU arms may differ in the last digits.
- **Timings.** Indicative only: the machine was shared with E1, and every
  row carries the one-minute load average. Clean timings are E6's job.

## Non-finite densities

pyvinecopulib's compiled vine returns a NaN density at a few test points of
some fits. These points are not floored: a floor at 1e-300 would add about
690 nats per point and turn a numerical fault into what looks like a
modeling failure. Instead, `e3_collect.py` scores every arm of an E3c or
E3d cell on the *common set*: the test points where every arm's density and
the truth's are finite, the same set for all arms within a design, d and
seed. `n_excluded` sits beside each row's own `n_nonfinite`.

Each arm's per-point log-densities stay on disk:
- E3c's npcc-side arrays hold the floor value where the density was NaN;
  the collector checks, in `mask_exact`, that those points are exactly the
  row's `n_nonfinite`;
- E3d's arrays and VDC's own-vine arrays are raw, NaN kept.

So the common set can always be recomputed. `e3_nan_diagnostics.py` writes
`results/e3_vdc/e3_nan_diagnostics.csv`, listing every row with a NaN and,
for the compiled vines, the tree and pair copula it comes from:

```bash
OMP_NUM_THREADS=1 taskset -c 14 nice -n 10 uv run --no-sync python scripts/paper/e3/e3_nan_diagnostics.py
```

## Deviations from VDC's published settings

- **E3a, TLL.** The TLL arm is pyvinecopulib's TLL itself, with its defaults
  (constant method, multiplier 1, a 30-point grid). These are the settings
  of VDC's `fit_bicop`; `e3a_vdc_side.py` asserts that and records both.
  VDC's runner instead chooses between independence and TLL by BIC; that is
  kept only as `tll_runner_bic`, the reproduction check.
- **E3a, n_train = 10,000.** Same generator and test set. The training draw
  uses the runner's seeds `[seed, seed + 1]` with `n = 10000`; it is not a
  superset of the 2,000-row draw (pyvinecopulib fills column by column).
- **E3b.** VDC's `e18_nonsimplified_stress` generator is not released, so it
  is rebuilt from the saved configuration and the appendix. On ten seeds it
  reproduces VDC's seed-0 numbers to within about 0.003 nats for VDC and
  parametric selection. The pooled TLL arm is TLL itself: on the pooled
  sign-flip sample the runner's BIC choice picks independence every time
  (TLL's effective degrees of freedom are about 130). Those rows are kept,
  labeled `excluded`.
- **E3c, designs.** The plan's "edges from the nine E3a cases" is the
  `strong` design: every tree as dependent as the first. There, sequential
  estimation loses most of its likelihood in the deep trees for every method,
  even well-specified parametric selection. The paper lead made the `decay`
  design primary (tau times 0.7 per tree, same orders, edge cases and
  seeds), and kept `strong` as a stress variant.
- **E3c.** VDC runs as a pyvinecopulib 1.0 pair copula, from PR #217's
  integration adapted to 1.0's API, so that every arm shares one vine
  machinery. VDC's own vine on the true order is kept as a reference. VDC's
  published vine timings compared TLL with R-vine selection on a CPU against
  VDC on a greedy D-vine on an H100. Here the structure is fixed and true
  for every arm, and every arm states its device.
- **PR #217's VDC numbers.** The integration loaded VDC 0.1.0 through a
  `sys.modules` shim that took `scatter_to_hist` from `vdc/data/hist.py`.
  That function returns cell probabilities summing to 1, where VDC's
  estimator expects a density with mean 1 — a factor of 64² too small.
  Making that substitution in v0.2.0 reproduces the PR's independence probe
  (53x at an anti-diagonal corner, mean |ĉ − 1| = 0.78). Without it, the
  same input gives 2.0x and 0.014.
