#!/usr/bin/env bash
# The weather application, configs/application.toml (scripts/paper/
# application/README.md), from a clean worktree W at the commit WANT, into
# the run folder OUT, in priority order, every unit resumable.
#
#   WANT=<sha> W=<worktree> OUT=<run folder> PY=<python> \
#     setsid -f nohup bash scripts/paper/application/launch.sh > <log> 2>&1
#
# Optional:
#   CONFIG       relative to W; configs/application.toml by default.
#   STATIONS     station ids, space-separated, over the configuration's
#                (every station): STATIONS=0 is the station-0 smoke.
#   RESUME       1 to continue a run folder a stopped launch left, every stage
#                with --resume; 0 by default, which refuses an existing OUT.
#   SETUP        a file sourced first, such as the cluster's module loads for
#                R; recorded in the launcher's settings.
#   GPU_SHARDS   GPU processes side by side, each on every GPU_SHARDS-th
#                station (1).
#   SHARDS       CPU processes side by side (2).
#   GPU_THREADS  each GPU process's BLAS and OpenMP threads (2).
#   GPU_CPUS, CPUS  taskset core lists for the GPU and the CPU processes;
#                empty (the default) pins nothing, as inside a Slurm job's
#                own cpuset.
#   MIN_FREE_GB  the free space the disk under OUT must keep (20).
#   STOP         a file whose existence stops the run before its next phase
#                (OUT.STOP by default).
#
# Stages, each after checking the stop file and the free space:
#   1. preflight;
#   2. margins: emos-seasonal in one CPU process, which also stores every
#      station's cases; then TabPFN and TabICL in the GPU processes beside
#      LightGBM on the CPU shards;
#   3. copulas: a GPU pipeline (the npcc arms with TabPFN or TabICL inside)
#      and a CPU pipeline (independence, gaussian, boostCopula, and the
#      chain with LightGBM inside) side by side, each in three phases:
#        a. tabpfn-quantiles-all, the triple: the main table;
#        b. tabpfn-quantiles-all's 5d arms, and emos-seasonal in both
#           designs: the 5d comparison;
#        c. everything else.
#      Every copula process runs with --resume, so a phase skips what an
#      earlier one, or an earlier launch, stored;
#   4. evaluate, and paired_summary.py into OUT/analysis.
# PYTORCH_CUDA_ALLOC_CONF is unset. Every command is appended to
# OUT/logs/provenance.txt as run; each process logs to OUT/logs/; this
# script's own record is OUT/logs/launcher.log, and it copies itself there.
set -uo pipefail
: "${WANT:?the commit to run}"
: "${W:?a clean worktree at WANT}"
: "${OUT:?the run folder}"
: "${PY:?the Python interpreter of the environment}"
CONFIG=${CONFIG:-configs/application.toml}
STATIONS=${STATIONS:-}
RESUME=${RESUME:-0}
SETUP=${SETUP:-}
GPU_SHARDS=${GPU_SHARDS:-1}
SHARDS=${SHARDS:-2}
GPU_THREADS=${GPU_THREADS:-2}
GPU_CPUS=${GPU_CPUS:-}
CPUS=${CPUS:-}
MIN_FREE_GB=${MIN_FREE_GB:-20}
STOP=${STOP:-$OUT.STOP}
SELF=$(readlink -f "$0")

GPU_MODELS=(tabpfn-quantiles-all tabpfn-quantiles-emos tabicl-all tabicl-emos)
CPU_MODELS=(lightgbm-all lightgbm-emos)
ALL_MODELS=(emos-seasonal lightgbm-all lightgbm-emos tabpfn-quantiles-all tabpfn-quantiles-emos tabicl-all tabicl-emos)
GPU_TRIPLE=(
  npcc-mixture-tabpfn-quantiles-all
  npcc-mixture-tabpfn-quantiles-time
  npcc-chain-tabpfn-quantiles-all
  npcc-chain-tabpfn-quantiles-time
  npcc-chain-tabicl-all
)
GPU_5D=(npcc-chain-tabpfn-quantiles-all@5d npcc-chain-tabicl-all@5d)
CPU_TRIPLE=(independence gaussian boostcopula-all boostcopula-time npcc-chain-lightgbm-all)
CPU_5D=(independence@5d gaussian@5d boostcopula-all@5d npcc-chain-lightgbm-all@5d)

if [ "$RESUME" = 1 ]; then
  [ -d "$OUT" ] || { echo "refused: RESUME=1 needs the run folder $OUT"; exit 1; }
  AGAIN=(--resume)
else
  [ -e "$OUT" ] && { echo "refused: $OUT exists; choose a fresh folder, or RESUME=1"; exit 1; }
  AGAIN=()
fi
if [ -n "$SETUP" ]; then
  # shellcheck source=/dev/null
  source "$SETUP" || { echo "refused: sourcing $SETUP failed"; exit 1; }
fi
cd "$W" || exit 1
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$WANT")" ] && [ -z "$(git status --porcelain)" ] || {
  echo "refused: $W must be a clean checkout of $WANT"
  exit 1
}
export PYTHONPATH=$W/src
got=$("$PY" -c "import npcc; print(npcc.__file__)")
[ "$got" = "$W/src/npcc/__init__.py" ] || { echo "refused: npcc imports from $got, not $W/src"; exit 1; }
unset PYTORCH_CUDA_ALLOC_CONF
mkdir -p "$OUT/logs"
L=$OUT/logs
cp "$SELF" "$L/launch.$(date +%Y%m%d-%H%M%S).sh"
log() { echo "$(date '+%F %T') $*" >> "$L/launcher.log"; }
render() { local s; printf -v s '%q ' "$@"; echo "(cd $W && PYTHONPATH=$W/src ${s% })"; }
pin() { [ -n "$1" ] && echo "taskset -c $1"; }
stopped() { [ -e "$STOP" ] && { log "stopped before $1: $STOP exists"; return 0; }; return 1; }
guard() {
  stopped "$1" && exit 0
  local free_kb
  free_kb=$(df -Pk "$OUT" | awk 'NR == 2 {print $4}')
  if [ "$free_kb" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
    log "DISK-STOP before $1: $((free_kb / 1024)) MB free under $OUT, below $MIN_FREE_GB GB"
    exit 1
  fi
  log "$1: $((free_kb / 1024 / 1024)) GB free"
}
APP=(-m npcc.experiments.application.cli)
# shellcheck disable=SC2206
PICK=(${STATIONS:+--stations $STATIONS})
# shellcheck disable=SC2046
CPU_ENV=(env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(pin "$CPUS") nice -n 15 "$PY")
# shellcheck disable=SC2046
GPU_ENV=(env OMP_NUM_THREADS="$GPU_THREADS" MKL_NUM_THREADS="$GPU_THREADS" OPENBLAS_NUM_THREADS="$GPU_THREADS" OMP_WAIT_POLICY=PASSIVE $(pin "$GPU_CPUS") nice -n 10 "$PY")
# Runs one command, recorded, its output appended to logs/<name>.log;
# returns its status.
run() {
  local name=$1; shift
  render "$@" >> "$L/provenance.txt"
  log "$name: start"
  "$@" >> "$L/$name.log" 2>&1 < /dev/null
  local rc=$?
  log "$name: exit $rc"
  return $rc
}
# Runs `count` processes of one command side by side, each on every
# count-th station; returns non-zero if any failed.
sharded() {
  local name=$1 count=$2; shift 2
  local -a pids=()
  local i rc=0
  for ((i = 0; i < count; i++)); do
    run "${name}_shard${i}of${count}" "$@" --shard "$i/$count" &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { log "${name}_shard${i}of${count} failed"; rc=1; }
  done
  return $rc
}
# One pipeline's three copula phases, in order; each process checks the disk
# before every unit and exits 3 below MIN_FREE_GB, that unit unwritten.
pipeline() {
  local kind=$1 count=$2; shift 2
  local -a env_=("$@") triple five
  if [ "$kind" = gpu ]; then triple=("${GPU_TRIPLE[@]}"); five=("${GPU_5D[@]}"); else triple=("${CPU_TRIPLE[@]}"); five=("${CPU_5D[@]}"); fi
  local base=(copulas --config "$CONFIG" --out "$OUT" "${PICK[@]}" --min-free-gb "$MIN_FREE_GB" --resume)
  stopped "${kind} phase a" && return 0
  sharded "copulas_${kind}_a" "$count" "${env_[@]}" "${APP[@]}" "${base[@]}" \
    --models tabpfn-quantiles-all --copulas "${triple[@]}" || return 1
  stopped "${kind} phase b" && return 0
  sharded "copulas_${kind}_b" "$count" "${env_[@]}" "${APP[@]}" "${base[@]}" \
    --models tabpfn-quantiles-all emos-seasonal --copulas "${triple[@]}" "${five[@]}" || return 1
  stopped "${kind} phase c" && return 0
  sharded "copulas_${kind}_c" "$count" "${env_[@]}" "${APP[@]}" "${base[@]}" \
    --models "${ALL_MODELS[@]}" --copulas "${triple[@]}" "${five[@]}" || return 1
}

log "start at $(git rev-parse --short=7 HEAD) on $(hostname): $CONFIG into $OUT (RESUME=$RESUME)"
echo "WANT=$WANT W=$W OUT=$OUT PY=$PY CONFIG=$CONFIG STATIONS=$STATIONS RESUME=$RESUME SETUP=$SETUP GPU_SHARDS=$GPU_SHARDS SHARDS=$SHARDS GPU_THREADS=$GPU_THREADS GPU_CPUS=$GPU_CPUS CPUS=$CPUS MIN_FREE_GB=$MIN_FREE_GB STOP=$STOP" >> "$L/launcher_settings.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "$L/launcher_settings.txt" 2>&1

guard "preflight"
run preflight "${CPU_ENV[@]}" "${APP[@]}" preflight --config "$CONFIG" "${PICK[@]}" || exit 1

guard "the EMOS margins"
run margins_emos "${CPU_ENV[@]}" "${APP[@]}" margins --config "$CONFIG" --out "$OUT" "${PICK[@]}" "${AGAIN[@]}" --models emos-seasonal || exit 1

guard "the npcc margins"
sharded margins_cpu "$SHARDS" "${CPU_ENV[@]}" "${APP[@]}" margins --config "$CONFIG" --out "$OUT" "${PICK[@]}" "${AGAIN[@]}" --models "${CPU_MODELS[@]}" &
cpu_margins=$!
sharded margins_gpu "$GPU_SHARDS" "${GPU_ENV[@]}" "${APP[@]}" margins --config "$CONFIG" --out "$OUT" "${PICK[@]}" "${AGAIN[@]}" --models "${GPU_MODELS[@]}"
rc=$?
wait "$cpu_margins" || rc=1
[ $rc -eq 0 ] || { log "margins failed"; exit 1; }

guard "the copula stage"
pipeline cpu "$SHARDS" "${CPU_ENV[@]}" &
cpu_pipeline=$!
pipeline gpu "$GPU_SHARDS" "${GPU_ENV[@]}"
rc=$?
wait "$cpu_pipeline" || rc=1
[ $rc -eq 0 ] || {
  log "copula stage stopped; a process exiting 3 found the disk below $MIN_FREE_GB GB (its log says which unit)"
  exit 1
}
stopped "evaluate" && exit 0
failed=$(find "$OUT/stations" -name failure.json | wc -l)
log "copula stage: $failed failed units"

guard "evaluate"
run evaluate "${CPU_ENV[@]}" "${APP[@]}" evaluate --config "$CONFIG" --out "$OUT" "${PICK[@]}" || exit 1
run paired_summary "${CPU_ENV[@]}" scripts/paper/application/paired_summary.py --run "$OUT" --out "$OUT/analysis" || exit 1
du -sh "$OUT" "$OUT/stations" > "$L/size.txt" 2>&1
log "all done"
