# Reproducing the paper's experiments

Every result in the paper and its supplement comes from committed code and
configs in this repository. This file lists, per experiment, what to run and
where the output goes, and the run the paper cites. It covers E1 to E4 and E6
to E10, with E1b, E1c, E2b, E2d, E3a to E3e, E7a, E7a′ and E9c; E5, the
application, is not here. The experiment plan itself lives with the paper
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

## Paper map

Where each table, figure and quoted number of the paper comes from, as the
paper's source comments name the folders, read at the paper's 80ebb89
("Restructure the paper around one story"). Folders are under `results/`, and
scripts under `scripts/paper/`, except `qt_ab/`: `scripts/qt_ab/` at the
commit named, on the `qt-ab-probit` branch (E7a). Each commit was read from
the folder itself: its `COMMIT`, `provenance.jsonl`, `logs/provenance.txt`,
or the commit every row records. "main" is this machine (16 vCPU, an NVIDIA
GRID A100D-20C vGPU profile with 20 GB); "gpu053" is gpu053.baobab (an A100
80GB PCIe). The section of each experiment below has its run record.

| Paper item | Exp. | Folders | Analysis | Commits | Machine |
|---|---|---|---|---|---|
| `tab:inner-model-accuracy`, `fig:accuracy-by-n`, `fig:conditional-tau` (§3.2); `tab:accuracy-by-n` (supp.) | E1, E1b | `study_2026-11-10`, with `study_2026-11-10_r` and `study_2026-11-10_r_smooth_cv` | `e1/analyze.py`, `e1/paired.py` | b7702e1 | main: 3 GPU shards; R on 8 CPU shards |
| `fig:misspecified` (§3.2), `tab:misspecified-by-n` (supp.) | E1c | `study_misspec_2026-11-10`, with `study_misspec_2026-11-10_r` | `e1/analyze.py`, `e1/paired.py` | b7702e1 | main: 2 GPU shards; R on 8 CPU shards |
| `tab:covariates` (§3.3); `tab:covariates-iae`, `-clayton`, `-gaussian` and the timings (supp.) | E9, E9c | `study_covariates_2026-11-11{,_r,_v35}`, `study_covariates_n10000_2026-11-11{,_r}` | `e9/analyze.py` | b74aee4; v3.5 run and analysis f8878a7; E9c dd605f2 | main |
| `tab:unconditional-pairs` (§3.4); `tab:non-simplified-pair`, `tab:vines-true-structure` (supp.) | E3a, E3b, E3c | `e3_vdc` | `e3/e3_collect.py` | npcc rows b7702e1; competitors c2f0cb7, d10182b, d18480b, per row | main: CPU, one core a fit; vGPU |
| `tab:unknown-structure` (§3.4); `tab:unknown-structure-catboost` (supp.) | E3d, E3e, E10 | `e3d_unknown_structure`, `e3e_simplification_2x2{,_gpu053}`, `e10_e3d{,_gpu053}_2026-11-11` | `e3/e3_collect.py`, `e3/e3e_report.py`, `e10/compare_e3d.py` | E3d's npcc rows b7702e1, competitors 2b319b7 and 4b83733; E3e and E10's TabPFN d9dc80e; E10's CPU arms as each row records | main; gpu053 for TabPFN at d = 10 `decay` |
| `tab:vine-results` (§3.5); `tab:vine-paired`, `tab:vine-copula`, `fig:vine-kl-by-x`, `fig:vine-kl-by-x-d10` (supp.): oracle, random, selected, kde-tll | E2, E2d | `vine_2026-11-10`, `vine_2026-11-11_selected`, `vine_n2000_2026-11-11`; at d = 10, `vine_d10_2026-11-11` and `vine_d10_n2000_2026-11-11` | `e2/analyze.py`, `e2/tree1_diagnostic.py` | b7702e1, f8878a7, b0523c5; at d = 10, b6cb9af and b0523c5, analyzed at 9b26ffa and 2f300e9 | main; gpu053 for d = 10 |
| the same tables: the mixture | E10 | `vine_study_mixture_2026-11-11`; `vine_study_mixture_d10_2026-11-11` (n = 1000 so far) | `e10/analyze_mixture.py`, `e10/analyze_mixture_gamvine.py` | d9dc80e | main; gpu053 for d = 10 |
| the same tables: the GAM vines; `app:gamvine` (supp.) | E2b | `vine_study_gamvine{,_d10,_n2000,_d10_n2000}_2026-11-11b`; the pilot `e2b_pilot_2026-11-11` | `e2/analyze.py` with `--with` | c6a876a; pilot cd312e2 | main: CPU, 6 single-threaded shards |
| `tab:cost` (§3.6) | E6; E2, E10 and E2b at d = 10 | `e6_timing` (`timings_summary.csv`, sections bivariate, e3a, e3d and vine); at d = 10, the `runtime_summary.parquet` of `vine_d10{,_n2000}_2026-11-11`, `vine_study_mixture_d10_2026-11-11` and `vine_study_gamvine_d10{,_n2000}_2026-11-11b` | `e6/consolidate.py` | b7702e1, the R rows 725ac8c (reused from `e6_timing_stopped_2026-11-06`); at d = 10, each run's | main, idle; at d = 10, gpu053, and main's CPU for the GAM vines |
| §3.7; `tab:coherence-directions`, `tab:coherence-projection` (supp.) | E4, E6 | `e4_coherence`; the projection's cost from `e6_timing` (section projection) | `e4/summarize.py`, `e6/consolidate.py` | b7702e1 | main: boosters on CPU, TabPFN on the vGPU; E6 idle |
| the finer projection grid (supp., "Coherence diagnostics") | E4 | `e4_projection_grid_converge` | `e4/summarize.py` | b7702e1 | main |
| `fig:density-recovery-icl`, `-gb` (supp.) | E1 | `study_2026-11-10/analysis/slices` | `e1/density_slices.py` | b7702e1 | main GPU |
| `app:competitors` (supp.) | E1b, E1c | `study_2026-11-10_r`, `study_2026-11-10_r_smooth_cv`, `study_misspec_2026-11-10_r` | `e1/r_selection.py`, `e1/cv_stops.py` | b7702e1 | main |
| `tab:releases-transforms` and the temperature ablation (supp.) | E1, E8 | `study_2026-11-10`, `study_2026-11-10_tabpfn_temperature` | `e1/analyze.py`, `e1/paired.py` | b7702e1 | main GPU |
| §2.2, the read-out; `app:readout-choice`, `tab:readout-choice` (supp.) | E7a′ | `qt_ab_probit_2026-11` | `qt_ab/report.py`, `qt_ab/readout_choice.py` | ca4e52c | main |
| the temperature A/B (supp.) | E8 | `qt_ab_probit_2026-11` (`tabpfn_temp1.0`, `report_temperature.txt`) | `qt_ab/report.py --compare-labels` | 42d8fda | main GPU |
| A.2, the interior and the read-out's bias (supp.) | E7a | `qt_ab_2026-11` (`exact_*.csv`) | `qt_ab/exact_*.py` | cb04041 | main CPU |
| A.2, Nori (supp.) | E7a′ | `nori_probe_probit_2026-11-11` | `qt_ab/nori_probe.py` | ca4e52c | main GPU |
| "Reusing quantile tables" (supp.) | E10 | `e10_table_reuse{,_e3d,_gpu053}_2026-11-11` | `e10/check_table_reuse.py` | a49337f; d9dc80e | main; gpu053 |

`results/qt_ab_2026-11` is a copy, made on 2026-11-12, of the E7a worktree's
`~/npcc-e7a/results/qt_ab_2026-11` (`COPIED_FROM.txt`); the two `exact_*.csv`
files the supplement cites are identical in both.

## Probit only (E8)

The paper presents one transformation of the copula scale, probit: the
normal scores behind TLL in (py)vinecopulib. Every paper config and script
therefore names E1's probit estimators, `<backend>-probit` and TabPFN's
`tabpfn-quantiles-<version>-probit`, with `tabpfn-quantiles-v3-probit` as
the main-text TabPFN. The library keeps logit, its default, and its other
transforms (top-level `README.md`, "Support transforms"); only the paper's
configs leave them out.

Every run the paper cites is probit: the probit rerun (TAG 2026-11-10, below)
and every run after it. The runs made before E8 fitted both transforms in E1
and logit only elsewhere; each section keeps their records under
"Superseded". E7a's commands stay on logit: they check out cb04041, whose
A/B scripts and labels are those of the run the supplement cites.

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
of them run at 1.0. The superseded runs from before E8 used the
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

A run from before E8 therefore cannot be analyzed, which is why the probit
rerun refitted the R competitors of E1b and E1c too: their scripts take the
draws as one more grid, but their fits are not stored. With the same R and
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

## E8 probit rerun (TAG 2026-11-10, b7702e1)

One orchestrated run refitted every npcc arm of E1, E1c, the temperature
ablation, E2, E3a to E3d and E4, and the density slices, under probit with
TabPFN at softmax temperature 1.0. It refitted E1b's and E1c's R competitors
for the Monte Carlo KL, collected and analyzed everything, then ran E6. It
ran from the main checkout at b7702e1, the merge of the temperature pull
request (#37), from 15:43 on 2026-11-10 to 04:23 on 2026-11-11. Each lane,
the finish and each corrective step first checked that `HEAD` was b7702e1
with `src`, `scripts`, `configs` and `r_methods` clean, that
`QuantileTableConfig().n_quantiles` was 21, and that
`scripts/paper/readout.py`'s `SINCE` was an ancestor of `HEAD`. Every
`provenance.jsonl` line in the run's folders records b7702e1 and no
uncommitted file.

- **Launchers.** In `results/rerun_2026-11-10/scripts/`, copied there on
  2026-11-12 from the session scratchpad where they ran (its `README.txt`):
  - `s3_run.sh`, the entry point (`Y=b7702e1 GRID=21 TAG=2026-11-10 setsid
    -f nohup bash s3_run.sh`), and `s3_env.sh`, which every script sources
    for the check above and the paths;
  - `s3_laneV.sh`, `s3_laneB.sh`, `s3_laneC.sh`, `s3_laneR.sh` and
    `s3_finish.sh`, the steps below;
  - `s3_outputs.py`, which moves the superseded outputs aside (`plan.txt`
    lists them), hashes the reused ones and checks them after;
  - `e6_by_hand.sh` and `cvstops_misspec_after_e9gpu.sh`, the corrective
    steps; `nori_after_e6*.sh` and `e9gpu_after_nori*.sh`, the runs queued
    after it; and `s3_after.sh`, `s3_e1agg.sh` and `watch_s3*.sh`.

  <!--
  codespell:ignore-begin
  -->
  `s3_run.sh` and every step queued after it log their start or exit to
  `gate.log` there.
  <!--
  codespell:ignore-end
  -->
  Each script sources `s3_env.sh` by that scratchpad path, so running one
  again means pointing it at a copy. The run's own log is
  `results/rerun_2026-11-10/logs/steps.log`, beside one log per step and
  `reused_sha256.json`.
- **Archive.** `s3_outputs.py archive` first moved the 2026-11-09 logit
  rerun's npcc outputs of E3a to E3d and E4, and its E6 run, into
  `results/superseded_2026-11-10/` (`MANIFEST.txt`, 100 entries), with a
  copy of each collector table the rerun would overwrite. `snapshot` then
  hashed 3816 reused files into `reused_sha256.json`.
- **Lanes**, run at once, lane R after lane C:

  | lane | what | how | when |
  |---|---|---|---|
  | V | TabPFN v3 vines: E3d `decay` d = 10, `random-ns` and `tll-ns`, ten seeds; E3d `strong` d = 10, `tll-ns`, seeds 101-505; E3c d = 10; E3d d = 5, `random-ns` and `tll-ns`; E3c d = 5; both designs where none is named | GPU, one process at a time | 15:43 to 03:05 |
  | B | E1 (3 shards, `--gpu-mem-fraction 0.2`); E1c (2 shards, 0.12) beside the temperature ablation (2 shards, 0.2); E2 (4 shards, 0.15); E3a's and E3b's GPU estimators (`e3_run_gpu_arms.py --main-tabpfn v3 --stages e3a e3b --no-collect`); E4's TabPFN runs; the density slices (4 parts, then the plot) | GPU, 4 threads a process, `OMP_WAIT_POLICY=PASSIVE`, `nice -n 10` | 15:43 to 23:50 |
  | C | the gradient boosters' E3a, E3b and E3c arms (CatBoost's with `rosenblatt-vinecop-check`), E3d's CatBoost `tll-ns`, `random-ns`, `random-s` and `true-s`, and their E4 runs: 21 jobs | CPU, 8 slots, one on each of cores 8-15, one thread | 15:43 to 16:16 |
  | R | E1b's gamCopula (`configs/study_r.toml --labels gamcopula`) and boostCopula-smooth-cv, then E1c's R competitors, each as 8 single-threaded shards and the aggregation pass | CPU, cores 8-15, `--device cpu` | 16:16 to 22:08 |

  No lane step failed. They wrote `study_2026-11-10`,
  `study_misspec_2026-11-10`, `study_2026-11-10_tabpfn_temperature`,
  `vine_2026-11-10`, `study_2026-11-10_r`, `study_2026-11-10_r_smooth_cv`
  and `study_misspec_2026-11-10_r`; the npcc cells of `e3_vdc` and
  `e3d_unknown_structure`; `e4_coherence`, `e4_projection_grid` and
  `e4_projection_grid_converge`; and `study_2026-11-10/analysis/slices`.
- **Finish** (`s3_finish.sh`, 03:05 to 03:10): `e3_collect.py`; E4's
  `summarize.py` on its three folders; E1's `analyze.py` and `paired.py`
  with the two E1b runs; the same for E1c with its R run; `r_selection.py` on
  each R run; the temperature ablation's three `paired.py` comparisons;
  E2's `analyze.py`; then `s3_outputs.py verify`, "all 3816 reused files
  unchanged". Each section below gives its commands.
- **Corrective steps.** `r_selection.py` failed on
  `study_2026-11-10_r_smooth_cv`: it reads gamCopula's selection lines,
  which boostCopula's fits do not write. With a failed step, the finish did
  not start E6. Each correction is in `steps.log` as `[finish] corrective`:
  - at 03:11, `cv_stops.py` on `study_2026-11-10_r_smooth_cv`, that
    folder's analysis in E1b (exit 0). `e6_by_hand.sh` ran it, and then E6,
    only because the failed step and E6's "not started" were the only
    failures in `steps.log`;
  - at 04:23, `cv_stops.py` on `study_misspec_2026-11-10_r`, whose
    boostCopula stops the supplement cites (exit 0), once E6, the Nori probe
    and E9's GPU pilot were over (`cvstops_misspec_after_e9gpu.sh`).
- **E6**, from 03:11 to 04:09 on the idle machine, with
  `E6_REUSE_R=results/e6_timing_stopped_2026-11-06`, into
  `results/e6_timing` (E6 below).
- **After E6**, queued from the same scripts but not part of the rerun: the
  Nori probe at ca4e52c (E7a′, 04:10) and E9's GPU pilot at 15c18c4 (04:11
  to 04:23; E9).
- **Unused fallback.** `s3_after.sh` would have retried a failed E1
  aggregation alone (`s3_e1agg.sh`); E1's aggregation did not fail, and it
  did nothing.
- **Reused, not rerun**, hashed before the lanes and unchanged after them:
  - E3's competitor rows, each with the commit it ran from
    (`scripts/paper/e3/README.md`): VDC, TLL and parametric in `e3_vdc`
    (E3a, E3b and both E3c designs), VDC's own vine and `table1_rerun_cpu`,
    and E3d's competitors in `e3d_unknown_structure`;
  - E6's R timings, `results/e6_timing_stopped_2026-11-06/r_*` (725ac8c);
  - the R runs from before E8 (`study_2026-11-05_r`, `_r_smooth`,
    `_r_smooth_cv` and `study_misspec_2026-11-06_r`), hashed so that nothing
    would change them; lane R refitted the arms the paper reports into the
    folders it cites.
- **No `COMMIT` or `logs/`** in `study_2026-11-10`, `study_misspec_2026-11-10`,
  `study_2026-11-10_tabpfn_temperature` and `vine_2026-11-10`: their commit
  and flags are in their `provenance.jsonl`, and their logs in the rerun's
  `logs/`. The R runs hold their shard logs in their own `logs/`.

### Superseded: the logit rerun (TAG 2026-11-09, 50d2ca5)

The same orchestration (`s2_*.sh`; lanes V, B and C, no lane R) from
50d2ca5, from 17:49 on 2026-11-09 to 09:53 on 2026-11-10. E1 fitted both
transforms and every other npcc arm logit, into `study_2026-11-09`,
`study_2026-11-09_tabpfn_temperature`, `study_misspec_2026-11-09` and
`vine_2026-11-09`. It had first moved the npcc outputs of E3 and E4 from
before E8 into `results/superseded_2026-11-06/`; the probit rerun then moved
its own, and its E6 run, into `results/superseded_2026-11-10/`. Its
launchers are in `results/rerun_2026-11-09/scripts/` and its logs in
`results/rerun_2026-11-09/logs/`. The paper cites none of it.

## E1 — bivariate backend study (§3.2)

- Config: `configs/study.toml` (8 estimators, all probit: the six backends
  and TabPFN's v2.5 and v3.5 releases, every TabPFN at softmax temperature
  1.0; 160 cells).
- Run: the sharded commands above with `OUT=results/study_<TAG>`.
- Analysis: `uv run --no-sync python scripts/paper/e1/analyze.py
  results/study_<TAG> --tabpfn <version> --transform probit` writes the
  tables (`overall.csv`, `by_n.csv`, `ranking*.csv`, `margins.csv`) and the
  figures (`accuracy_by_n`, `conditional_tau`, each as PDF and PNG) into
  `results/study_<TAG>/analysis/`. With E1b's runs after `--with`, it adds
  the `*_with_competitors` figures.
- Refitting some estimators after a code change: `npcc-simstudy --config
  configs/study.toml --labels <label> ... --out <fresh folder>` fits only
  those estimators, each exactly as in the full run (same data, same torch
  seed); every shard and the aggregation pass take the same `--labels`.
  `analyze.py` and `paired.py` then take `--replacing <fresh folder>`, whose
  rows replace those labels' rows of the other runs; pass `--out` as well,
  so the original analysis stays as it was.
- Density slices (supplement): `scripts/paper/e1/density_slices.py fit
  --tabpfn <version> --transform probit --out
  results/study_<TAG>/analysis/slices` (add `--part I/K` to split), then
  `... density_slices.py plot --out ...`. The refits use E1's cell and
  estimator seeds, so the models equal E1's up to GPU nondeterminism.
- E1's density metrics are unprojected; the projection results reported
  come from E4.
- TabPFN temperature ablation (supplement):
  `configs/study_tabpfn_temperature.toml` (v3 and v3.5 at 0.9, against E1's
  1.0, probit; E1's cells for n in {500, 2000}, reps 0-4) with the sharded
  commands above. Its fits pair with E1's on identical data, `E1` and `T`
  being the two runs' folders: `scripts/paper/e1/paired.py $E1 --with $T
  --labels tabpfn-quantiles-v3-temp0.9-probit --against
  tabpfn-quantiles-v3-probit --common-cells --out $T/analysis/v3`, and
  likewise `tabpfn-quantiles-v3.5-temp0.9-probit` against
  `tabpfn-quantiles-v3.5-probit` (into `analysis/v3.5`) and against
  `tabpfn-quantiles-v3-temp0.9-probit` (into `analysis/vs_v3`).
- **Ran** in the probit rerun (above), from b7702e1, with TAG 2026-11-10:
  - `results/study_2026-11-10`: 3 GPU shards (`--workers 1 --resume
    --shard i/3 --gpu-mem-fraction 0.2`, 4 threads each) from 15:43 on
    2026-11-10, then the aggregation pass at 18:08;
  - the analysis, in the rerun's finish, into
    `results/study_2026-11-10/analysis/`:

    ```bash
    E1=scripts/paper/e1
    R=results
    uv run --no-sync python $E1/analyze.py $R/study_2026-11-10 \
        --with $R/study_2026-11-10_r $R/study_2026-11-10_r_smooth_cv \
        --tabpfn v3 --transform probit
    uv run --no-sync python $E1/paired.py $R/study_2026-11-10 \
        --with $R/study_2026-11-10_r $R/study_2026-11-10_r_smooth_cv \
        --labels tabpfn-quantiles-v3-probit tabicl-probit best-gb \
        --against gamcopula boostcopula-smooth-cv
    ```

  - the density slices, `--tabpfn v3`, as 4 parts and the plot, into
    `results/study_2026-11-10/analysis/slices`; all 288,000 rows record
    b7702e1;
  - the temperature ablation, 2 GPU shards (`--gpu-mem-fraction 0.2`) into
    `results/study_2026-11-10_tabpfn_temperature`, and in the finish its
    three `paired.py` comparisons above into that folder's `analysis/`.
- **Cited**, and frozen:
  - `study_2026-11-10/analysis`: `overall.csv`, `by_n.csv`, `ranking.csv`
    and `paired.csv` (`tab:inner-model-accuracy`, the supplement's
    `tab:accuracy-by-n` and `tab:releases-transforms`, and §3.2's numbers),
    `accuracy_by_n_with_competitors.pdf` (`fig:accuracy-by-n`, copied to
    the paper's `figures/accuracy_by_n.pdf`) and `conditional_tau.pdf`
    (`fig:conditional-tau`);
  - `study_2026-11-10/analysis/slices`: the supplement's density figures,
    `density_recovery_grid_icl.pdf` and `density_recovery_grid_gb.pdf`;
  - `study_2026-11-10_tabpfn_temperature/analysis/{v3,v3.5,vs_v3}/paired.csv`:
    the supplement's temperature ablation.

### Superseded

- `results/study_2026-11-05`, from before E8, on both transforms and the
  checkpoints' temperatures: the four shards ran once each from 2f8fa94, and
  the aggregation pass from 7000ae3 only read their checkpoints. Its
  `projection_summary` predates the midpoint Sinkhorn projection (a26262c).
  Its density slices record 7000ae3 for parts 0–2 and d18480b for part 3,
  rerun after a CUDA launch failure (nothing E1 runs changed in between).
- `results/study_2026-11-06_tabpfn_temperature`, from 7567717, before E8:
  the temperature ablation with the checkpoints' temperatures swapped
  instead, v3 at 1.0 and v3.5 at 0.9.
- `results/study_2026-11-09` and `results/study_2026-11-09_tabpfn_temperature`,
  the logit rerun (TAG 2026-11-09, above).

## E1b — established competitors on E1's cells (§3.2)

R 4.6.2 with VineCopula 2.6.1 and mgcv 1.9-4; boostCopula 0.1.0 at commit
746b191 of jobstdavid/boostCopula; gamCopula 0.0-8 built from commit e5561a8
of tvatter/gamCopula (PR #20, which passes a fitted t copula's second
parameter to its density and distribution). Each config's `[grid]` table is
E1's, so cells, seeds and truths are E1's, and each script's SHA-256 enters
its estimator id, so a run's signature pins the script it ran. All runs use
the sharded commands above with `--device cpu`; the runs the paper cites ran
as 8 single-threaded shards (`--shard I/8`, with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at 1).

- `configs/study_r.toml`: gamCopula, and boostCopula linear in x. The paper
  reports its gamCopula arm (`--labels gamcopula`).
- `configs/study_r_smooth_cv.toml`: boostCopula on a cubic B-spline basis of
  x, stopped by 10-fold cross-validation. This is the boostCopula arm
  reported.
- Analysis: E1's `analyze.py` with these runs after `--with` adds the
  `*_with_competitors` figures, and `scripts/paper/e1/r_selection.py <run>`
  joins each gamCopula fit's family, formula and EDF with its accuracy into
  `<run>/analysis/r_selection.csv`. It reads gamCopula's selection lines in
  the shard logs, so it fails on a run of boostCopula alone.
- CV stops: `scripts/paper/e1/cv_stops.py <run>` writes each
  cross-validated fit's stop out of the cap (`cv_stops.csv`) and, per n, how
  many stopped within 5% of it (`cv_stops_by_n.csv`), into the run's
  `analysis/`.
- Paired comparison: `scripts/paper/e1/paired.py <run> --with <R run>
  --labels tabpfn-quantiles-v3-probit tabicl-probit best-gb --against gamcopula
  boostcopula-smooth-cv` writes each estimator's pdf KL and IAE minus each
  competitor's, cell by cell, as mean, SE and wins over repetitions (overall,
  by family and by n) into `<run>/analysis/paired.csv`.
- **Ran** in the probit rerun's lane R (above), from b7702e1, with the shard
  logs of each run in its `logs/`. The fits log the package versions above.
  - `configs/study_r.toml --labels gamcopula` into
    `results/study_2026-11-10_r`, from 16:16 on 2026-11-10;
  - `configs/study_r_smooth_cv.toml` into
    `results/study_2026-11-10_r_smooth_cv`, from 16:18 to 20:10;
  - in the finish, `r_selection.py results/study_2026-11-10_r` (the same on
    the boostCopula run failed, as it must), then, as a corrective step,
    `cv_stops.py results/study_2026-11-10_r_smooth_cv`; E1's `analyze.py` and
    `paired.py` with both runs (E1).
- **Cited**, and frozen: `study_2026-11-10_r/analysis/r_selection.csv` and
  `study_2026-11-10_r_smooth_cv/analysis/cv_stops_by_n.csv` (the
  supplement's "Parametric competitors" and §3.2's gamCopula numbers at
  n = 200); the competitors' rows of E1's tables and figures and of
  `study_2026-11-10/analysis/paired.csv`.

### Superseded

From before E8, on the grid KL; their fits are not stored, so they cannot
give the Monte Carlo KL:
- `results/study_2026-11-05_r`, from 349ec08: `configs/study_r.toml`, both
  arms.
- `results/study_2026-11-05_r_smooth_cv`, from d10182b, as 8
  single-threaded shards. The one line of its `provenance.jsonl` is a later
  aggregation pass, from 9b809b2.
- `results/study_2026-11-05_r_smooth`, from b8f26dd:
  `configs/study_r_smooth.toml`, the same basis stopped by AIC, kept for the
  record and not rerun under E8. It stops at maxit = 500 in 63 of 160 fits.

## E1c — misspecified scenarios (§3.2)

Two shape families with a closed-form tau(x) of their own, under the
`implied` regime: a Clayton-to-Gumbel mixture with Gumbel weight x (M1) and a
Student t whose degrees of freedom run from 2 to 50 (M2); 80 cells.

- npcc arms (TabPFN v3, TabICL, Nori, LightGBM, XGBoost, CatBoost):
  `configs/study_misspec.toml`, as GPU shards and the aggregation pass.
- R competitors (gamCopula, boostCopula with CV stopping):
  `configs/study_misspec_r.toml`, as 8 single-threaded CPU shards (`--shard
  I/8 --device cpu`, with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `MKL_NUM_THREADS` at 1).
- Analysis: `analyze.py <npcc run> --with <R run>`, `paired.py` with the
  same runs and labels as in E1b, and `cv_stops.py <R run>` for its
  boostCopula stops.
- **Ran** in the probit rerun (above), from b7702e1:
  - `results/study_misspec_2026-11-10`, lane B: 2 GPU shards
    (`--gpu-mem-fraction 0.12`) from 18:08 on 2026-11-10, beside the
    temperature ablation, then the aggregation pass at 19:19;
  - `results/study_misspec_2026-11-10_r`, lane R: 8 single-threaded shards
    from 20:10 to 22:08;
  - in the finish, `analyze.py results/study_misspec_2026-11-10 --with
    results/study_misspec_2026-11-10_r --tabpfn v3 --transform probit`,
    `paired.py` and `r_selection.py`; then, at 04:23 on 2026-11-11, as a
    corrective step, `cv_stops.py results/study_misspec_2026-11-10_r`.
- **Cited**, and frozen: `study_misspec_2026-11-10/analysis` (`overall.csv`,
  `by_n.csv`, `paired.csv`, `ranking.csv` and
  `accuracy_by_n_with_competitors.pdf`, which is `fig:misspecified`, copied
  to the paper's `figures/misspecified_by_n.pdf`; the supplement's
  `tab:misspecified-by-n`), and
  `study_misspec_2026-11-10_r/analysis/cv_stops_by_n.csv`.

### Superseded

- `results/study_misspec_2026-11-06` and `..._r`, from a clean worktree at
  7567717, before E8: the npcc arms on logit as 2 GPU shards, the R
  competitors as 8 single-threaded shards.
- `results/study_misspec_2026-11-09`, the logit rerun (TAG 2026-11-09,
  above).

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
- E9's pilots, `results/e9_pilot_2026-11-11`: not cited, and to be
  archived. They ran on 2026-11-11, before E9's pull request was merged,
  from the E9 branch's worktree at three commits, so their provenance is
  mixed:
  - `r/`, from b9c2c7c: the R arms on every family and scenario at n =
    2000, rep 0 (six cells; the scenario `constant` still named `null`),
    plus boostCopula at `df = 5` from a copy of its script beside the
    config, for timings; 6 single-threaded shards from 01:38;
  - `r_iters100/`, from b9c2c7c: the two gamCopula scripts at `n.iters =
    100`, from copies beside the config, on the same cells;
  - `r_oracle_basis/`, from 9e1d433: the oracle with gamBiCopVarSel's basis
    rule, from a copy of the script beside the config, made before the rule
    was committed. Its three `provenance.jsonl` lines record the oracle
    script as uncommitted, as `_methods/fit_gamcopula_oracle.R`: the
    recorder drops the first character of the first path. The copy equals
    `r_methods/fit_gamcopula_oracle.R` at 15c18c4 and at b74aee4;
  - `gpu/`, from 15c18c4, clean: the npcc arms on one repetition of every
    family, scenario and n (12 cells), one GPU process, from 04:11 to 04:23
    after the probit rerun's E6, for the GPU-hour estimate
    (`results/rerun_2026-11-10/scripts/e9gpu_after_nori_w.sh`). Its
    `configs`, `r_methods` and `src` equal b74aee4's.
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
- Analysis: `uv run --no-sync python scripts/paper/e2/analyze.py
  results/vine_<TAG>` writes the table (`vine_table.csv`, in the columns
  of `tab:vine-results`), the figure (`vine_kl_by_x`, PDF and PNG, with the
  plotted values in `vine_kl_by_x.csv`) and the numbers the text quotes
  (`vine_text_numbers.json`) into `results/vine_<TAG>/analysis/`, and
  prints the table's LaTeX rows. Timings come from E6.
- **Ran** in the probit rerun (above), lane B: `results/vine_2026-11-10`,
  from b7702e1, as 4 shards (`--shard i/4 --resume --workers 1
  --gpu-mem-fraction 0.15`) started at 19:20 on 2026-11-10, then the
  aggregation pass at 20:50.
  - **No `logs/` or `COMMIT`.** This run has neither. Its commit and flags
    come from the `provenance.jsonl` the runner wrote, one line per
    process, each recording b7702e1 and no uncommitted file.
  - **Analysis.** The rerun's finish ran the command above on that folder,
    which wrote `results/vine_2026-11-10/analysis/`.
  - **Cited**, and frozen: its `vine_table.csv` and `vine_text_numbers.json`
    give the oracle, random and kde-tll entries of `tab:vine-results` and
    the supplement's `tab:vine-paired` at d = 3 and 5, n = 500 and 1000, and
    §3.5's numbers there; its `vine_kl_by_x.pdf` is the supplement's
    `fig:vine-kl-by-x`.
- **Superseded:**
  - `results/vine_2026-11-06`, before E8, on logit, from a clean worktree
    at d10182b: 4 shards of `npcc-vinestudy` with the flags above on two GPU
    slots, then the aggregation pass; `logs/provenance.txt` has the exact
    commands. A shard that ran out of GPU memory was resumed, which reuses
    its finished cells: a cell's seeds depend on the cell alone. A d = 5
    shard peaks at about 7 GB, since every fitted margin held its own TabPFN
    model on the GPU before f4f17b5.
  - `results/vine_2026-11-09`, the logit rerun (TAG 2026-11-09, above).
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
    for bit. The pilot is `results/e2_selected_pilot_2026-11-11`, from a
    clean worktree at 0059940, the arm's commit on its branch, as one GPU
    process; its `logs/provenance.txt` has the command. It is not cited.
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
    - its `vine_selected.csv` gives the selected arm's entries of
      `tab:vine-results` and `tab:vine-paired` at d = 3 and 5;
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
    - `vine_selected.csv` gives the selected arm's rows there;
    - `tree1_diagnostic.csv` gives the tree-1 numbers at d = 10;
    - `vine_kl_by_x.pdf` is the supplement's `fig:vine-kl-by-x-d10`, copied to
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
    npcc-vinestudy --config <config> --device cpu --out <run> --shard i/k
    --resume` for i below k, then once without `--shard`; the runs below
    used k = 6), with R, gamCopula, mgcv and VineCopula installed.
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
  - **Ran** on 2026-11-11 and 12, here on the CPU, from a clean worktree at
    c6a876a, the tip of E2b's pull request (#50) with the igraph fix. The
    four configs ran in turn, each as 6 single-threaded shards on cores 4
    to 9 and then the aggregation pass, into `results/<config>_2026-11-11b`.
    No fit failed in any of them.

    | config | cells | shards |
    |---|---|---|
    | `vine_study_gamvine` | 80 | 20:18 to 20:43 |
    | `vine_study_gamvine_d10` | 40 | 20:43 to 22:58 |
    | `vine_study_gamvine_n2000` | 40 | 22:58 to 23:18 |
    | `vine_study_gamvine_d10_n2000` | 20 | 23:18 to 01:02 |

    Each run's analysis is `analyze.py` on the E2 run it pairs with, with
    the GAM run after `--with`, written into the GAM run's own `analysis/`,
    which leaves the E2 runs' cited `analysis/` folders as they were:

    ```bash
    A=scripts/paper/e2/analyze.py
    R=results
    uv run --no-sync python $A $R/vine_2026-11-10 --with $R/vine_2026-11-11_selected \
        $R/vine_study_gamvine_2026-11-11b --out $R/vine_study_gamvine_2026-11-11b/analysis
    uv run --no-sync python $A $R/vine_d10_2026-11-11 --with $R/vine_study_gamvine_d10_2026-11-11b \
        --out $R/vine_study_gamvine_d10_2026-11-11b/analysis
    uv run --no-sync python $A $R/vine_n2000_2026-11-11 --with $R/vine_study_gamvine_n2000_2026-11-11b \
        --out $R/vine_study_gamvine_n2000_2026-11-11b/analysis
    uv run --no-sync python $A $R/vine_d10_n2000_2026-11-11 \
        --with $R/vine_study_gamvine_d10_n2000_2026-11-11b \
        --out $R/vine_study_gamvine_d10_n2000_2026-11-11b/analysis
    ```

    Each run's `logs/` holds the launcher (`e2b.sh`), `provenance.txt`
    with every command, R's and the packages' versions, and each shard's
    log. A first run from fa3b844, before the igraph fix, went into
    `results/vine_study_gamvine_2026-11-11` and `..._d10_2026-11-11`; it
    was stopped and superseded, and each holds a `SUPERSEDED.txt` saying
    why. Do not cite it.

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
  - **Ran** on 2026-11-11 from b0523c5, the tip of E2d's pull request
    (#48):
    - d = 3 and 5, here, from a clean worktree, as one GPU process
      (`npcc-vinestudy --config configs/vine_study_n2000.toml --out
      results/vine_n2000_2026-11-11 --workers 1 --resume --gpu-mem-fraction
      0.9`, 4 threads), 17:22 to 18:44 for the 40 cells. `analyze.py` and
      `tree1_diagnostic.py` followed from the same worktree.
    - d = 10, on gpu053 from a checkout at b0523c5: a pilot of one cell
      (rep 0) into `vine_d10_n2000_pilot_2026-11-11` in 21 min, then the
      run into `vine_d10_n2000_2026-11-11`, one process with the same
      flags, 17:58 to 22:10 for the 20 cells. Both were copied unchanged
      into `results/` and analyzed here on the CPU from 2f300e9, the base
      after #48 and #49.

    Each run's `logs/` holds its launcher, `provenance.txt` with every
    command, and `steps.log`. E10's analysis (below) pairs both runs with
    E2's cell by cell and refuses any that do not pair; it accepted both.

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
- **Against the GAM vines.** `scripts/paper/e10/analyze_mixture_gamvine.py
  <run> --with <E2b runs> --out <dir>` reads the mixture against E2b's three
  GAM arms and writes `vine_mixture_gamvine.csv`, one row per (d, n, GAM
  arm). For the joint and the copula KL, each averaged over x within a
  repetition first, it gives the paired difference mixture - GAM arm as mean
  and SE over repetitions, and the repetitions in which the mixture is
  lower. It reuses `analyze_mixture.py`'s loading and pairing, so it refuses
  a `--with` run that does not hold the same cell seed and true log-density
  at every (d, n, rep, x) both runs hold, or that shares no cell with the
  mixture run: pass the E2b runs at the mixture run's d and n only. `--out`
  has no default, since the mixture runs' `analysis/` folders are cited:

  ```bash
  A=scripts/paper/e10/analyze_mixture_gamvine.py
  R=results
  uv run --no-sync python $A $R/vine_study_mixture_2026-11-11 \
      --with $R/vine_study_gamvine_2026-11-11b $R/vine_study_gamvine_n2000_2026-11-11b \
      --out <dir>
  uv run --no-sync python $A $R/vine_study_mixture_d10_2026-11-11 \
      --with $R/vine_study_gamvine_d10_2026-11-11b --out <dir>
  ```

  At d = 10 the pairing crosses machines: the mixture ran on gpu053 and the
  GAM vines here, on the same cells, whose truths and test points the check
  finds equal bit for bit. While the d = 10 mixture run holds n = 1000 only,
  `vine_study_gamvine_d10_n2000_2026-11-11b` shares no cell with it and is
  refused. On 2026-11-12 both commands, from this branch, reproduced an
  ad hoc computation from the same `metrics_by_x.parquet` files to the
  fourth decimal, every mean, SE and count.
- **Reusing quantile tables.** A pair copula in a vine reads each direction
  twice at the same conditioning rows, for its density and then for its
  h-function. Each `ConditionalMargin` now keeps its last table and returns
  it for a repeat read. `scripts/paper/e10/check_table_reuse.py` checks that
  this changes no TabPFN vine density bitwise, on one E2 cell and one E3d
  sample, and measures the speed-up. It times three passes, off, on and off,
  each starting with no table kept.
  - On E2's cell d = 5, n = 1000, rep 0 (random arm, 5,000 points), from
    a49337f into `results/e10_table_reuse_2026-11-11`: 52.8 s, 31.7 s and
    52.7 s, 1.67x, joint and copula densities bitwise equal. That folder's
    E3d timing is void: before 444ea50, each pass reused the tables the
    previous pass had kept.
  - On E3d's `decay`, d = 5, seed 101 (random structure, non-simplified,
    20,000 points), from d9dc80e into
    `results/e10_table_reuse_e3d_2026-11-11` (`--skip-e2 --e3d-folder
    results/e3d_unknown_structure`): 93.2 s, 61.0 s and 95.0 s, 1.54x,
    bitwise equal.
  - On gpu053, E3d's `decay`, d = 10, seed 101 (2,000 points), from d9dc80e
    into `results/e10_table_reuse_gpu053_2026-11-11`: 37.3 s, 23.5 s and
    37.2 s, 1.58x, bitwise equal.
- **Ran (d = 3 and 5)** here, from a clean worktree at d9dc80e, the tip of
  E10's pull request (#51) before the base was merged into it, as one GPU
  process (4 threads, `--workers 1 --resume --gpu-mem-fraction 0.9`) into
  `results/vine_study_mixture_2026-11-11`. It ran in two passes: n = 500
  and 1000 from 22:59 on 2026-11-11 to 01:33, then n = 2000, resuming the
  first pass's cells, to 03:04, for 120 cells. The analysis:

  ```bash
  R=results
  uv run --no-sync python scripts/paper/e10/analyze_mixture.py $R/vine_study_mixture_2026-11-11 \
      --with $R/vine_2026-11-10 $R/vine_2026-11-11_selected $R/vine_n2000_2026-11-11
  ```

  It wrote `analysis/` with `vine_mixture.csv`, `vine_mixture_cells.csv` and
  `vine_mixture_checks.json`: Jensen's bound holds in every cell (to
  3e-17), and member 1 equals the selected arm bit for bit. The first
  pass's own analysis named the n = 2000 run, which had no cell in common
  with it yet, and was refused; its log is `logs/analyze_mixture_pass1.log`.
  `logs/` also holds the launcher, `vine_two_pass.sh`, and
  `provenance.txt`.
- **Ran (d = 10)** on gpu053, as a whole experiment, from a checkout at
  d9dc80e, into `vine_study_mixture_d10_2026-11-11`: one sample size at a
  time, n = 1000, then 2000, then 500, each as one process
  (`--workers 1 --resume --gpu-mem-fraction 0.9`, 4 threads) on a copy of
  the config narrowed to that n (`config_n<n>.toml` in the run's folder).
  Each n is analyzed as it finishes, against gpu053's own E2 runs at
  d = 10, into `analysis_after_n1000`, then `analysis_after_n1000_2000`,
  and `analysis` when all three are in. n = 1000 ran from 05:45 to 09:19 on
  2026-11-12. The run's `logs/` holds the queue (`gpu053_queue_v2.sh`),
  its log, and `provenance.txt`, and `env_gpu053.txt` records the machine.
  The folder is copied unchanged into `results/`.

## E3 — comparison with VDC (§3.4)

See `scripts/paper/e3/README.md` for VDC's environment, the weight check and
the command behind every file in `results/e3_vdc/`. It also covers E3d,
E3e (`results/e3e_simplification_2x2/`) and E10 on E3d's truths
(`results/e10_e3d_2026-11-11/`). Its "The runs the paper cites" gives the
commit and run behind every row: the NPCC arms of E3a to E3d are the probit
rerun's (above), and the competitors' rows are reused from before E8.

## E4 — numerical coherence (§3.7)

- Accuracy and coherence, as two runs that differ only in estimators and
  device; every row records the commit it ran from.
  - The gradient boosters on CPU: `uv run --no-sync python
    scripts/paper/e4/coherence.py --labels lightgbm-probit xgboost-probit
    catboost-probit --n 500 2000 --reps 5 --sinkhorn 5 20 50 --device cpu
    --part I/3 --out results/e4_coherence` for I in 0, 1, 2.
  - TabPFN on the GPU: the same with `--labels tabpfn-quantiles-v3-probit`,
    no `--device`, `--part I/2` for I in 0, 1, and `--out
    results/e4_coherence/tabpfn`.
- Projection-grid ablation: the same script with `--projection-grid 30 60`
  scores the projection on both grid sizes from one fit per cell, at n = 2000
  on E4's 20 cells (`--n 2000 --reps 5`), for LightGBM on CPU and TabPFN v3
  on the GPU: `--sinkhorn 20` into `results/e4_projection_grid/<backend>`,
  and `--sinkhorn 20 50 100` into
  `results/e4_projection_grid_converge/<backend>`, with `<backend>` one of
  `lightgbm` and `tabpfn`. Each folder's summary adds `projection_grid.csv`,
  `projection_grid_change.csv` (the paired change from unprojected) and
  `projection_grid_cost.csv`; the `proj_*` columns are the margin errors on
  the projection grid itself.
- Projection cost per distinct conditioning value, on an otherwise idle
  device: the same script with `--cost-only`, which E6 runs.
- Summary: `uv run --no-sync python scripts/paper/e4/summarize.py
  results/e4_coherence`, which reads the parts in both folders, and the
  same on each projection-grid folder.
- **Ran** in the probit rerun (above), from b7702e1, which every row of
  every part records:
  - lane C, each process on one core with one thread: the boosters' three
    coherence parts and LightGBM's two projection-grid runs, from 16:03 to
    16:16 on 2026-11-10;
  - lane B, on the vGPU: TabPFN's two coherence parts from 22:54, then its
    two projection-grid runs from 23:19;
  - the finish: `summarize.py` on `results/e4_coherence`,
    `results/e4_projection_grid` and `results/e4_projection_grid_converge`.
- **Cited**, and frozen:
  - `e4_coherence/{directional,hfunc,projection,mass_tail}.csv`: §3.7 and
    the supplement's `tab:coherence-directions` and
    `tab:coherence-projection`, the projection rows at 20 iterations;
  - `e4_projection_grid_converge/projection_grid_change.csv`: the finer
    projection grid in the supplement's "Coherence diagnostics", at 100
    iterations.

  `results/e4_projection_grid` is not cited:
  `results/e4_projection_grid_converge` scores 20 iterations too.

### Superseded

- From before E8, on logit, moved by the logit rerun into
  `results/superseded_2026-11-06/`, each part recording its commit:
  - `e4_coherence`: the boosters from 7909ed7, TabPFN from 7000ae3. E1c's
    scenario code (2b319b7) was in the working tree but uncommitted; it
    reproduces 7000ae3's seeds, draws and truths for these cells bit for
    bit.
  - `e4_projection_grid`: LightGBM from 4b83733, TabPFN from 7567717.
  - `e4_projection_grid_converge`: TabPFN from 7a3150a, LightGBM from
    2cc08f7.
- The logit rerun's three folders (TAG 2026-11-09, 50d2ca5), moved by the
  probit rerun into `results/superseded_2026-11-10/`.

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

- **Ran** at the end of the probit rerun (above), from the main checkout at
  b7702e1, into `results/e6_timing`, on 2026-11-11 from 03:11 to 04:09.
  `e6_by_hand.sh` started it once the machine was idle (no GPU process,
  load below 1.5), with `E6_REUSE_R=results/e6_timing_stopped_2026-11-06`.
  The run's `logs/provenance.txt` records b7702e1, no uncommitted file and
  that folder; `logs/steps.log` has each job's start, and `logs/` each
  job's log. In `timings.csv`, the 64 rows of the R competitors record
  725ac8c and every other row b7702e1.
- **The R timings** come from `results/e6_timing_stopped_2026-11-06`, an E6
  run from 725ac8c on 2026-11-06, stopped during its vine job: its
  `r_gamcopula` and `r_boostcopula_cv` jobs ran from 06:47 to 07:27, one
  thread each. Between 725ac8c and b7702e1, neither R script, neither
  config, nor `npcc.experiments._external` changed.
- **Cited**, and frozen: `timings_summary.csv`, sections bivariate, e3a
  and e3d (from `e3_timing.csv`) and vine (d <= 5) in §3.6's `tab:cost`,
  and projection in §3.7.
- **Superseded:** the logit rerun's E6 (TAG 2026-11-09, 50d2ca5, with the
  same R timings; its E3 step rerun at 8dc5761), moved by the probit rerun
  into `results/superseded_2026-11-10/e6_timing_2026-11-09`; and that E3
  step's first attempt, which failed at 04:13 on 2026-11-10, in
  `results/superseded_2026-11-06/e6_timing_e3_failed_2026-11-10`.

## E7a — choice of the read-out (supplement)

The A/B that chose the quantile read-out, and its checks on exact quantiles,
run from cb04041. Their scripts live in that commit's `scripts/qt_ab/`, whose
README describes each one, and every output row records the commit. The
exact-quantile checks and the tree and TabPFN runs come first (2026-11-09);
the TabICLv2 and Nori run and `nori_probe.py` need the whole GPU and follow
E6. Check the commit out to reproduce its four outputs into
`results/qt_ab_2026-11/`. The supplement now cites two of them, the
exact-quantile checks; E7a′ (below) supersedes the A/B and the Nori probe.

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

- **Ran** from ca4e52c into `results/qt_ab_probit_2026-11` (`COMMIT`:
  ca4e52c, clean; `events.log` has the timeline), on 2026-11-10 from 11:22
  to 12:04. Three cells of the in-context run, written while the worktree
  was being edited under the running process, recorded a dirty tree or
  bed40f8; they were set aside in `icl_misstamped/` and rerun from ca4e52c,
  so every row of `gb/`, `tabpfn/` and `icl/` records ca4e52c. The
  supplement cites `readout_choice.csv` (`tab:readout-choice`, reps 5-9)
  and `report_relative_selection.txt` (the selection, reps 0-4); §2.2 names
  the read-out it chose.
- **Nori under probit** (the supplement's A.2): `scripts/qt_ab/nori_probe.py`
  at ca4e52c, on E1's cell clayton/linear, rep 0, at n = 200 and 2000, both
  transforms. It ran on 2026-11-11 at 04:10, after the probit rerun's E6, from
  a clean detached worktree at ca4e52c with `~/npcc/.venv`'s interpreter and
  `PYTHONPATH` on the worktree's `src`, with `QT_AB_RESULTS` set to
  `results/nori_probe_probit_2026-11-11`. That folder holds `COMMIT`
  (ca4e52c, clean), `env.txt`, `nori_probe.log` and `nori_probe.csv`, whose
  rows record ca4e52c; the supplement cites its probit rows. The launcher is
  `results/rerun_2026-11-10/scripts/nori_after_e6_w.sh`.

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

**Ran** from 42d8fda on 2026-11-10, from 14:13 to 14:26, into
`results/qt_ab_probit_2026-11/tabpfn_temp1.0`, whose rows all record
42d8fda; `report_temperature.txt` beside it is the supplement's source.
E1's temperature ablation, `results/study_2026-11-10_tabpfn_temperature`,
is a separate run (E1).

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
- **Ran** on 2026-11-11 here, with `WANT=a42716a TAG=2026-11-11`, the tip
  of the screen's pull request (#49), into `results/qnorm_screen_2026-11-11`:
  E1's 80 cells from 18:51 to 19:16, E3a's 180 fits to 20:03, so 1 h 12 min
  in all.
- **Verdict: keep `u`.** `report.txt` closes with "KEEP u: gain overall
  (z = -10.43), but loss beyond 2 SE in family frank". The overall score is
  -0.00196 (SE 0.00019), and every n gains beyond 2 SE. The Frank family
  (E3a's Frank copula at theta = 5, both `n_train`) loses: +0.00091 (SE
  0.00016, z = +5.7), with `qnorm(u)` worse in all 10 of its pairs. By the
  rule, `u` is kept.
