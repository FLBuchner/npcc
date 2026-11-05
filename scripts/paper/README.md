# Reproducing the paper's experiments

Every result in the paper and its supplement comes from committed code and
configs in this repository. This file lists, per experiment, what to run and
where the output goes. The experiment plan itself (E1–E7) lives with the paper
(`paper/jcgs/experiments.md` in the paper repository).

All commands run from the repository root, with the environment synced as

```bash
uv sync --frozen --extra cu130 --extra backends --extra experiments
```

(pick the CUDA flavor that matches the driver). Results are written under
`results/`, which is not tracked; each output records the npcc commit it was
produced from.

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

## E1 — bivariate backend study (§3.2)

- Config: `configs/study.toml` (16 estimators; 160 cells).
- Run: the sharded commands above with `OUT=results/study_2026-11-05`.
- Analysis: `uv run --no-sync python scripts/paper/e1/analyze.py
  results/study_2026-11-05 --tabpfn <version> --transform <transform>` writes
  the tables (`overall.csv`, `by_n.csv`, `ranking*.csv`, `margins.csv`) and the
  figures (`accuracy_by_n`, `conditional_tau`, each as PDF and PNG) into
  `results/study_2026-11-05/analysis/`.
- Density slices (supplement): `scripts/paper/e1/density_slices.py fit
  --tabpfn <version> --transform <transform> --out
  results/study_2026-11-05/analysis/slices` (add `--part I/K` to split), then
  `... density_slices.py plot --out ...`. The refits use E1's cell and
  estimator seeds, so the models equal E1's up to GPU nondeterminism.

- TabPFN temperature ablation (supplement): `configs/study_tabpfn_temperature.toml`
  (v3 at temperature 1.0 and v3.5 at 0.9, logit; E1's cells for n in
  {500, 2000}, reps 0-4) with the sharded commands above and
  `OUT=results/study_2026-11-06_tabpfn_temperature`; its fits pair with E1's
  on identical data, so `analyze.py results/study_2026-11-05 --with
  results/study_2026-11-06_tabpfn_temperature` puts both in one table.

## E3 — comparison with VDC (§3.3)

See `scripts/paper/e3/README.md` for VDC's environment, the weight check and
the command behind every file in `results/e3_vdc/`.

## E4 — numerical coherence (§3.5)

- Accuracy and coherence, as two runs that differ only in estimators and
  device; every row records the commit it ran from.
  - The gradient boosters on CPU, from 7909ed7: `uv run --no-sync python
    scripts/paper/e4/coherence.py --labels lightgbm-logit xgboost-logit
    catboost-logit --n 500 2000 --reps 5 --sinkhorn 5 20 50 --device cpu
    --part I/3 --out results/e4_coherence` for I in 0, 1, 2.
  - TabPFN on the GPU, from 7000ae3: the same with `--labels
    tabpfn-quantiles-v3-logit`, no `--device`, `--part I/2` for I in 0, 1,
    and `--out results/e4_coherence/tabpfn`. E1c's scenario code (2b319b7)
    was in the working tree but uncommitted; it reproduces 7000ae3's seeds,
    draws and truths for these cells bit for bit.
- Projection cost per distinct conditioning value, on an otherwise idle
  device: the same script with `--cost-only`.
- Summary: `uv run --no-sync python scripts/paper/e4/summarize.py
  results/e4_coherence`, which reads the parts in both folders.

## Still to be filled in

E1b (gamCopula and boostCopula on E1's cells), E2 (the vine rerun) and E6
(clean timings) are added here when their runs are final.
