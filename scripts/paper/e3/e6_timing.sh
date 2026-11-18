#!/usr/bin/env bash
# E6, the §3.3 part: clean timings of E3a's pair copulas and E3d's vines.
#
# E6's driver runs it on an idle machine, after its other parts:
#
#     bash scripts/paper/e3/e6_timing.sh [results/e6_timing]
#
# (scripts/paper/e6/run_all.sh calls it with its own output folder). One job
# runs at a time. Everything goes to <out>/e3/: e3a/ (E3a's own scripts,
# re-run there), e3d/ (the vine timings) and logs/ (one log per step, and
# steps.log with each step's start). Last it writes <out>/e3_timing.csv,
# which E6's consolidate.py reads. That file has one row per section, method,
# setting, d, n, seed and stage, with device, compute, CPU model, thread
# count and npcc commit. The script refuses to start if <out>/e3/ exists,
# since E3a's scripts resume and would skip the cells timed earlier.
#
# Protocol:
# - CPU methods run on one core (E6_CPU_CORE): VDC, TLL, parametric, and
#   NPCC with LightGBM, XGBoost or CatBoost. Every BLAS and OpenMP pool,
#   and torch, has one thread.
# - GPU methods (NPCC with TabPFN v3, TabICL or Nori) run on the GPU with
#   E6_GPU_THREADS host cores and threads (default 2, as in E3's GPU runs).
# - Every timing follows an untimed warm-up in the same process: each
#   script's own for E3a, and e6_timing.py's and e6_timing_vdc.py's for the
#   vines.
# - E3a, n = 2000: every method on the same nine samples, seed 202 of each
#   case (E6_E3A_SEEDS). The VDC side fits all 45 samples in order, so the
#   first case's seed 101 also warms up TLL and parametric.
# - E3d's vines, on E3d's samples, decay design:
#   - TabPFN v3 with arm (a), a random structure, non-simplified, at
#     d = 5 (seeds 101, 202, 303) and d = 10 (seed 101);
#   - TLL, parametric and VDC with and without structure selection, at
#     d = 5 and d = 10 (seeds 101, 202, 303).
#
# Expected wall time on an idle machine: about 55 minutes. The TabPFN vine
# at d = 10 takes about 25 of them, almost all inference.
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT=${1:-results/e6_timing}
E3=$OUT/e3
S=scripts/paper/e3
VDC_PYTHON=/home/thibault/vdc/env/bin/python
CPU_CORE=${E6_CPU_CORE:-4}
GPU_THREADS=${E6_GPU_THREADS:-2}
GPU_CORES=${E6_GPU_CORES:-4-$((4 + GPU_THREADS - 1))}
E3A_SEEDS=${E6_E3A_SEEDS:-202}
VINE_SEEDS=${E6_VINE_SEEDS:-101 202 303}
TABPFN_D10_SEEDS=${E6_TABPFN_D10_SEEDS:-101}

if [ -e "$E3" ]; then
  echo "e6_timing.sh: $E3 must not exist yet; move it aside first" >&2
  exit 1
fi
mkdir -p "$E3/logs"

# E3a's fit index: nine cases, each with seeds 101, 202, 303, 404 and 505.
E3A_ONLY=""
for seed in $E3A_SEEDS; do
  case $seed in
    101) j=0 ;; 202) j=1 ;; 303) j=2 ;; 404) j=3 ;; 505) j=4 ;;
    *) echo "e6_timing.sh: E3a seeds must be 101, 202, 303, 404 or 505; got $seed" >&2; exit 1 ;;
  esac
  for i in 0 1 2 3 4 5 6 7 8; do E3A_ONLY="$E3A_ONLY $((5 * i + j))"; done
done

unset PYTORCH_CUDA_ALLOC_CONF  # expandable_segments fails on this machine's vGPU
step() { echo "$(date +%H:%M:%S) $*" | tee -a "$E3/logs/steps.log"; }
on_cpu() {
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c "$CPU_CORE" "$@"
}
on_gpu() {
  OMP_NUM_THREADS=$GPU_THREADS MKL_NUM_THREADS=$GPU_THREADS OPENBLAS_NUM_THREADS=$GPU_THREADS \
    taskset -c "$GPU_CORES" "$@"
}
# An array rather than a function: on_cpu and on_gpu run it through taskset,
# which executes a program and cannot see a shell function.
NPCC=(uv run --no-sync python)

step "start: CPU core $CPU_CORE; GPU methods on cores $GPU_CORES; E3a seeds $E3A_SEEDS (fits$E3A_ONLY)"

step "E3a: VDC, TLL and parametric (VDC environment, one core)"
on_cpu "$VDC_PYTHON" $S/e3a_vdc_side.py --n-train 2000 \
  --reference-run results/e3_vdc/table1_rerun_cpu --out "$E3/e3a" > "$E3/logs/e3a_vdc_side.log" 2>&1

step "E3a: NPCC with LightGBM, XGBoost and CatBoost (one core)"
on_cpu "${NPCC[@]}" $S/e3a_npcc_fit.py --folder "$E3/e3a" --n-train 2000 --device cpu \
  --estimators lightgbm-probit xgboost-probit catboost-probit --only $E3A_ONLY \
  > "$E3/logs/e3a_npcc_cpu.log" 2>&1

step "E3a: NPCC with TabPFN v3, TabICL and Nori (GPU)"
on_gpu "${NPCC[@]}" $S/e3a_npcc_fit.py --folder "$E3/e3a" --n-train 2000 --device cuda --gpu-is-free \
  --estimators tabpfn-quantiles-v3-probit tabicl-probit nori-probit --only $E3A_ONLY \
  > "$E3/logs/e3a_npcc_gpu.log" 2>&1

step "E3d: TLL and parametric, with and without selection (one core)"
on_cpu "${NPCC[@]}" $S/e6_timing.py vines --part competitors --dims 5 10 --seeds $VINE_SEEDS \
  --out "$E3/e3d" > "$E3/logs/e3d_competitors.log" 2>&1

step "E3d: VDC, with and without its order selection (VDC environment, one core)"
on_cpu "$VDC_PYTHON" $S/e6_timing_vdc.py --dims 5 10 --seeds $VINE_SEEDS \
  --out "$E3/e3d" > "$E3/logs/e3d_vdc.log" 2>&1

step "E3d: NPCC with TabPFN v3, arm (a), d = 5 (GPU)"
on_gpu "${NPCC[@]}" $S/e6_timing.py vines --part npcc --device cuda --gpu-is-free --dims 5 \
  --seeds $VINE_SEEDS --out "$E3/e3d" > "$E3/logs/e3d_npcc_d5.log" 2>&1

step "E3d: NPCC with TabPFN v3, arm (a), d = 10 (GPU)"
on_gpu "${NPCC[@]}" $S/e6_timing.py vines --part npcc --device cuda --gpu-is-free --dims 10 \
  --seeds $TABPFN_D10_SEEDS --out "$E3/e3d" > "$E3/logs/e3d_npcc_d10.log" 2>&1

step "collect"
"${NPCC[@]}" $S/e6_timing.py collect --root "$OUT" --e3a-seeds $E3A_SEEDS 2>&1 | tee "$E3/logs/collect.log"
step "done"
