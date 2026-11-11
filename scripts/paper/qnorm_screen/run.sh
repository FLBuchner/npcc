#!/usr/bin/env bash
# The qnorm screen (scripts/paper/README.md): TabPFN v3 handed qnorm(u)
# against u, on E1's cells at reps 5-9 and on E3a's nine copulas, then the
# verdict of the rule fixed in report.py.
#   WANT=<sha> TAG=<tag> setsid -f nohup bash <worktree>/scripts/paper/qnorm_screen/run.sh > <log> 2>&1
# Runs from the clean worktree holding this script, which must be at WANT,
# with ~/npcc/.venv's interpreter and PYTHONPATH on the worktree's src, so
# every row records the worktree's commit. Writes into a new folder,
# $RES/qnorm_screen_$TAG:
#   e1/        npcc-simstudy on configs/qnorm_screen.toml, both arms;
#   e3a/       e3a_npcc_fit.py --out, both arms, reading E3a's samples from
#              $RES/e3_vdc/e3a, which it leaves untouched;
#   analysis/  report.py's pairs.csv, groups.csv and report.txt;
#   COMMIT, env.txt, logs/ (this launcher, provenance.txt with every command
#   rendered, steps.log, one log per step).
# The GPU part is one GPU process at a time (--workers 1, 4 threads on cores
# 0-3, passive waits, nice 10; two processes on this vGPU crash with CUDA
# errors): the E1 run, then the E3a fits. A failed step is logged FAILED and
# stops the run. Each step resumes from its own checkpoints, so after a
# failure its command in provenance.txt can be rerun as it stands.
set -uo pipefail
: "${WANT:?the commit to run from: the pushed tip of the screen PR}"
: "${TAG:?the date tag of the run}"
W=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
RES=${RES:-/home/thibault/npcc/results}
PY=${PY:-/home/thibault/npcc/.venv/bin/python}
OUT=$RES/qnorm_screen_$TAG
LOG=$OUT/logs
BASE=tabpfn-quantiles-v3-probit
ARM=tabpfn-quantiles-v3-qnorm-probit
unset PYTORCH_CUDA_ALLOC_CONF  # expandable_segments fails on this vGPU
export PYTHONPATH=$W/src

# Refusals before anything is written, so a refused start leaves no folder.
die() { echo "REFUSED: $*" >&2; exit 1; }
[ -e "$OUT" ] && die "$OUT already exists"
cd "$W" || die "no worktree at $W"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$WANT")" ] \
  || die "the worktree must be at $WANT; got $(git rev-parse --short=7 HEAD)"
[ -z "$(git status --porcelain)" ] || die "the worktree must be clean"
got=$("$PY" -c "import npcc; print(npcc.__file__)")
[ "$got" = "$W/src/npcc/__init__.py" ] \
  || die "npcc must import from $W/src; got $got"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
  || die "the GPU must be free"

mkdir -p "$LOG"
cp "$(readlink -f "$0")" "$LOG/run.sh"
LANE=screen
step() { echo "$(date +%T) [$LANE] $*" >> "$LOG/steps.log"; }
stop() { LANE=screen; step "stopped: $*"; exit 1; }

step "start at $(git rev-parse --short=7 HEAD) (pid $$) into $OUT"
{
  echo "HEAD $(git rev-parse HEAD)"
  echo "worktree $W"
  echo "status (empty = clean):"
  git status --porcelain
} > "$OUT/COMMIT"
{
  uv pip freeze --python "$PY"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  grep -m1 "model name" /proc/cpuinfo
} > "$OUT/env.txt" 2>&1

GPU_ENV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  OMP_WAIT_POLICY=PASSIVE taskset -c 0-3 nice -n 10)
E1_CMD=("${GPU_ENV[@]}" "$PY" -m npcc.experiments
  --config configs/qnorm_screen.toml --out "$OUT/e1" --workers 1 --resume
  --gpu-mem-fraction 0.9)
E3A_CMD=("${GPU_ENV[@]}" "$PY" scripts/paper/e3/e3a_npcc_fit.py
  --folder "$RES/e3_vdc/e3a" --out "$OUT/e3a"
  --config configs/qnorm_screen.toml --estimators "$BASE" "$ARM"
  --n-train 2000 10000 --device cuda --gpu-is-free)
REPORT_CMD=("$PY" scripts/paper/qnorm_screen/report.py --e1 "$OUT/e1"
  --e3a "$OUT/e3a" --out "$OUT/analysis")

render() {
  local s
  printf -v s '%q ' "$@"
  echo "(cd $W && PYTHONPATH=$W/src ${s% })"
}
{
  echo "# the qnorm screen at $(git rev-parse HEAD); launcher: logs/run.sh"
  echo "# E1's cells at reps 5-9, both arms: one GPU process, which aggregates at its end"
  render "${E1_CMD[@]}"
  echo "# E3a's nine copulas at n_train 2000 and 10000, both arms: one GPU process"
  render "${E3A_CMD[@]}"
  echo "# the verdict, on the CPU; its output is analysis/report.txt"
  render "${REPORT_CMD[@]}"
} > "$LOG/provenance.txt"

LANE=e1
step "E1's cells, reps 5-9, $BASE and $ARM: one GPU process (cores 0-3) into $OUT/e1"
"${E1_CMD[@]}" > "$LOG/e1.log" 2>&1 < /dev/null \
  || stop "FAILED: the E1 run (see logs/e1.log)"
step "done ($(find "$OUT/e1/cells" -name DONE | wc -l) cells)"

LANE=e3a
step "E3a's nine copulas, n_train 2000 and 10000, $BASE and $ARM: one GPU process (cores 0-3) into $OUT/e3a"
"${E3A_CMD[@]}" > "$LOG/e3a.log" 2>&1 < /dev/null \
  || stop "FAILED: the E3a fits (see logs/e3a.log)"
step "done ($(find "$OUT/e3a/npcc_cells" -name 'k*.json' | wc -l) fits)"

LANE=report
step "scripts/paper/qnorm_screen/report.py"
mkdir -p "$OUT/analysis"
"${REPORT_CMD[@]}" > "$OUT/analysis/report.txt" 2> "$LOG/report.log" \
  || stop "FAILED: the report (see logs/report.log)"
step "$(grep -- '-- verdict' "$OUT/analysis/report.txt")"
LANE=screen
step "all done"
