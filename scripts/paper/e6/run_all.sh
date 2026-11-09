#!/usr/bin/env bash
# E6: every clean timing, one job at a time on an otherwise idle machine.
#
#   bash scripts/paper/e6/run_all.sh [results/e6_timing]
#
# Run from the repository root at a committed state. It refuses to start
# while another process holds the GPU or the one-minute load average is 2 or
# more, and into an output folder that already exists. Each step logs to
# <out>/logs/; the last one writes timings.csv and timings_summary.csv.
#
# E6_REUSE_R=<earlier E6 folder> copies that run's R competitor timings
# (r_gamcopula/, r_boostcopula_cv/) instead of rerunning them: they time the
# R scripts alone, which no npcc change touches, and each keeps the commit
# its own provenance.jsonl records.
set -euo pipefail
OUT=${1:-results/e6_timing}
PY=(uv run --no-sync python)
R_METHODS=(gamcopula boostcopula_cv)

if [ -n "${E6_REUSE_R:-}" ]; then
  for r in "${R_METHODS[@]}"; do
    if [ ! -f "$E6_REUSE_R/r_$r/runtime.parquet" ]; then
      echo "E6_REUSE_R must hold r_$r/runtime.parquet; got none in $E6_REUSE_R" >&2
      exit 1
    fi
  done
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "E6 needs an idle GPU; another process holds it" >&2
  exit 1
fi
load=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$load" 'BEGIN { exit !(l >= 2) }'; then
  echo "E6 needs an idle machine; the load average is $load" >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  echo "E6 writes a fresh folder; $OUT already exists" >&2
  exit 1
fi

mkdir -p "$OUT/logs"
{
  echo "npcc_commit $(git rev-parse HEAD)"
  echo "uncommitted $(git status --porcelain src configs scripts r_methods | wc -l)"
  echo "started $(date -Is)"
  echo "r_reused_from ${E6_REUSE_R:-none}"
} > "$OUT/logs/provenance.txt"
step() { echo "== $1 $(date -Is)" | tee -a "$OUT/logs/steps.log"; }
one_thread=(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)

step "bivariate, boosters on all cores"
"${PY[@]}" scripts/paper/e6/timing.py bivariate --out "$OUT" \
  > "$OUT/logs/bivariate.log" 2>&1

step "bivariate, boosters on one thread"
env "${one_thread[@]}" "${PY[@]}" scripts/paper/e6/timing.py bivariate \
  --labels lightgbm-logit xgboost-logit catboost-logit --gb-threads 1 \
  --tag gb1 --out "$OUT" > "$OUT/logs/bivariate_gb1.log" 2>&1

for r in "${R_METHODS[@]}"; do
  if [ -n "${E6_REUSE_R:-}" ]; then
    step "R competitor $r, reused from $E6_REUSE_R"
    cp -r "$E6_REUSE_R/r_$r" "$OUT/r_$r"
    continue
  fi
  step "R competitor $r, one thread"
  env CUDA_VISIBLE_DEVICES="" "${one_thread[@]}" uv run --no-sync \
    npcc-simstudy --config "configs/e6_timing_$r.toml" --out "$OUT/r_$r" \
    --device cpu --workers 1 > "$OUT/logs/r_$r.log" 2>&1
done

step "vines"
"${PY[@]}" scripts/paper/e6/timing.py vine --out "$OUT" \
  > "$OUT/logs/vines.log" 2>&1

step "projection cost, TabPFN v3"
"${PY[@]}" scripts/paper/e4/coherence.py --labels tabpfn-quantiles-v3-logit \
  --n 2000 --reps 1 --sinkhorn 20 --cost-only --projection-grid 30 60 \
  --cost-repeats 3 --out "$OUT/projection_cost/tabpfn" \
  > "$OUT/logs/projection_tabpfn.log" 2>&1

step "projection cost, LightGBM"
"${PY[@]}" scripts/paper/e4/coherence.py --labels lightgbm-logit \
  --n 2000 --reps 1 --sinkhorn 20 --cost-only --projection-grid 30 60 \
  --cost-repeats 5 --device cpu --out "$OUT/projection_cost/lightgbm" \
  > "$OUT/logs/projection_lightgbm.log" 2>&1

if [ -f scripts/paper/e3/e6_timing.sh ]; then
  step "unconditional comparison (E3)"
  bash scripts/paper/e3/e6_timing.sh "$OUT" > "$OUT/logs/e3.log" 2>&1
else
  step "unconditional comparison skipped: no scripts/paper/e3/e6_timing.sh"
fi

step "consolidate"
"${PY[@]}" scripts/paper/e6/consolidate.py "$OUT"
step "done"
