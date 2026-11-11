# E3 — comparison with VDC (§3.3)

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

Two environments are involved. VDC's code runs only in its own (`~/vdc/env`,
Python 3.10, pyvinecopulib 0.7.5). Everything else runs in npcc's (`uv run
--no-sync python ...`, pyvinecopulib 1.0). Nothing from VDC is installed into
npcc's `.venv`. Where an npcc-side fit needs VDC's estimate — the adapted VDC
pair copula of E3a's cross-check and of E3c — it gets it from
`e3_vdc_server.py`, a process in VDC's environment.

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
| `e3d_npcc_fit.py` | npcc | E3d `parametric` and `tll` with Dissmann selection, true-structure references, and NPCC on random, TLL-selected and true structures |
| `e3_vdc_server.py` | VDC | VDC's density estimate as a service (started by `e3_vdc_pair.VdcServer`) |
| `e3_vdc_pair.py` | npcc | The server client and `VdcBicop`: VDC's 64 x 64 grid replicate-padded and held by pyvinecopulib 1.0's `TorchTllBicop` with `norm_maxiter=0`, adapted from pyvinecopulib PR #217 |
| `e3_vine.py` | npcc | `SharedVinecop`: a simplified pyvinecopulib 1.0 `VinecopBase` hosting any torch pair copula |
| `e3_npcc.py` | npcc | E1's estimators, at the probit transform (same labels and specs), and one timed fit |
| `e3_collect.py` | npcc | Long tables, summaries, cross-checks, the common-set scoring, and the code check |
| `e3_nan_diagnostics.py` | npcc | Which arms, trees and pair copulas give non-finite vine densities |
| `e3_run_gpu_arms.py` | npcc | The pending GPU arms, once the GPU is free |
| `e6_timing.sh` | both | E6's §3.3 part: clean timings of E3a's pairs and E3d's vines, one job at a time |
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
`/home/thibault/...` paths). Edit them there to run anywhere else.

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
run recorded before E8 used the same commands with E1's logit estimators.

Run everything from the npcc repository root. The CPU runs pin each process
to one core and one BLAS/OpenMP thread, as VDC's runner times its methods;
those are the settings below, with `nice -n 10` because the machine was
shared. The core numbers are what the recorded run used and only matter for
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
For the primary `decay` design (`results/e3_vdc/e3c_decay/`; the recorded
run used cores 10–13):

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

The recorded run of E3a, E3b and the strong design ran as five chains, one
per core (11–15), starting each npcc-side command once its VDC side had
written `vdc_side.json`. An npcc-side script needs nothing else, so the order
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

### E6's timings (§3.3)

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
