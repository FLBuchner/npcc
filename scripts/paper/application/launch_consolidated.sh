#!/usr/bin/env bash
# The consolidated application run, configs/application_consolidated.toml
# (scripts/paper/application/README.md, "The consolidated design"), from a
# clean worktree W at the commit WANT, into the run folder OUT.
#
#   WANT=<sha> W=<worktree> OUT=<run folder> PY=<python> \
#     setsid -f nohup bash scripts/paper/application/launch_consolidated.sh \
#     > <log> 2>&1
#
# Optional: CONFIG (relative to W; the consolidated config by default),
# SHARDS (CPU shards, 6), CPUS (their cores, 4-11), GPU_CPUS (the GPU
# process's cores, 0-3), MIN_FREE_GB (2), STOP (a file whose existence stops
# the run before its next stage; OUT.STOP by default).
#
# Stages, each after checking the stop file and that the disk under OUT
# holds at least MIN_FREE_GB free:
#   1. margins of emos-seasonal, one CPU process, which also stores every
#      station's cases, so no two later processes write them;
#   2. margins of TabPFN and TabICL, one GPU process, beside LightGBM's on
#      SHARDS CPU shards;
#   3. the copula stage: its npcc arms in one GPU process, beside its
#      classical arms on SHARDS CPU shards;
#   4. evaluate, and paired_summary.py into OUT/analysis.
# One GPU process at a time: a GPU stage waits until no other process holds
# the GPU. PYTORCH_CUDA_ALLOC_CONF is unset. Every command is appended to
# OUT/logs/provenance.txt as run; each process logs to OUT/logs/; this
# script's own record is OUT/logs/launcher.log, and it copies itself there.
# npcc-application writes OUT's COMMIT, env.txt, manifest.json, config.json
# and provenance.jsonl.
set -uo pipefail
: "${WANT:?the commit to run}"
: "${W:?a clean worktree at WANT}"
: "${OUT:?the run folder, which must not exist}"
: "${PY:?the Python interpreter of the environment}"
CONFIG=${CONFIG:-configs/application_consolidated.toml}
SHARDS=${SHARDS:-6}
CPUS=${CPUS:-4-11}
GPU_CPUS=${GPU_CPUS:-0-3}
MIN_FREE_GB=${MIN_FREE_GB:-2}
STOP=${STOP:-$OUT.STOP}
SELF=$(readlink -f "$0")

GPU_MODELS=(tabpfn-quantiles-all tabpfn-quantiles-emos tabicl-all tabicl-emos)
CPU_MODELS=(lightgbm-all lightgbm-emos)
NPCC_ARMS=(
  npcc-mixture-tabpfn-quantiles-all
  npcc-mixture-tabpfn-quantiles-time
  npcc-chain-tabpfn-quantiles-all
  npcc-chain-tabpfn-quantiles-time
  npcc-chain-tabpfn-quantiles-all@5d
)
CLASSICAL_ARMS=(
  independence
  gaussian
  boostcopula-all
  boostcopula-time
  independence@5d
  gaussian@5d
)

[ -e "$OUT" ] && { echo "refused: $OUT exists; choose a fresh folder"; exit 1; }
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
cp "$SELF" "$L/"
log() { echo "$(date '+%F %T') $*" >> "$L/launcher.log"; }
render() { local s; printf -v s '%q ' "$@"; echo "(cd $W && PYTHONPATH=$W/src ${s% })"; }
gpu_free() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done; }
guard() {
  if [ -e "$STOP" ]; then log "stopped before $1: $STOP exists"; exit 0; fi
  local free_kb
  free_kb=$(df -Pk "$OUT" | awk 'NR == 2 {print $4}')
  if [ "$free_kb" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
    log "DISK-STOP before $1: $((free_kb / 1024)) MB free under $OUT, below $MIN_FREE_GB GB"
    exit 1
  fi
  log "$1: $((free_kb / 1024 / 1024)) GB free"
}
APP=(-m npcc.experiments.application.cli)
CPU_ENV=(env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c "$CPUS" nice -n 15 "$PY")
GPU_ENV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE taskset -c "$GPU_CPUS" nice -n 10 "$PY")
# Runs one command, recorded, its output in logs/<name>.log; returns its status.
run() {
  local name=$1; shift
  render "$@" >> "$L/provenance.txt"
  log "$name: start"
  "$@" > "$L/$name.log" 2>&1 < /dev/null
  local rc=$?
  log "$name: exit $rc"
  return $rc
}
# Runs SHARDS CPU processes of one stage and the given GPU command beside
# them; returns non-zero if any failed.
beside() {
  local stage=$1 gpu_name=$2; shift 2
  local -a pids=() names=()
  local i
  for ((i = 0; i < SHARDS; i++)); do
    run "${stage}_cpu_shard${i}of${SHARDS}" "${CPU_ENV[@]}" "${APP[@]}" "${CPU_ARGS[@]}" --shard "$i/$SHARDS" &
    pids+=($!)
    names+=("${stage}_cpu_shard${i}of${SHARDS}")
  done
  gpu_free
  run "$gpu_name" "$@"
  local rc=$?
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { log "${names[$i]} failed"; rc=1; }
  done
  return $rc
}

log "start at $(git rev-parse --short=7 HEAD) on $(hostname): $CONFIG into $OUT"
echo "WANT=$WANT W=$W OUT=$OUT PY=$PY CONFIG=$CONFIG SHARDS=$SHARDS CPUS=$CPUS GPU_CPUS=$GPU_CPUS MIN_FREE_GB=$MIN_FREE_GB STOP=$STOP" > "$L/launcher_settings.txt"

guard "preflight"
run preflight "${CPU_ENV[@]}" "${APP[@]}" preflight --config "$CONFIG" || exit 1

guard "the EMOS margins"
run margins_emos "${CPU_ENV[@]}" "${APP[@]}" margins --config "$CONFIG" --out "$OUT" --models emos-seasonal || exit 1

guard "the npcc margins"
CPU_ARGS=(margins --config "$CONFIG" --out "$OUT" --models "${CPU_MODELS[@]}")
beside margins margins_gpu "${GPU_ENV[@]}" "${APP[@]}" margins --config "$CONFIG" --out "$OUT" --models "${GPU_MODELS[@]}" || exit 1

guard "the copula stage"
# Each copula process also checks the disk before every unit and exits with
# status 3, that unit unwritten, below MIN_FREE_GB: the same command with
# --resume then continues from there once space is freed.
CPU_ARGS=(copulas --config "$CONFIG" --out "$OUT" --min-free-gb "$MIN_FREE_GB" --copulas "${CLASSICAL_ARMS[@]}")
beside copulas copulas_gpu "${GPU_ENV[@]}" "${APP[@]}" copulas --config "$CONFIG" --out "$OUT" --min-free-gb "$MIN_FREE_GB" --copulas "${NPCC_ARMS[@]}" || {
  log "copula stage stopped; a process exiting 3 found the disk below $MIN_FREE_GB GB (its log says which unit)"
  exit 1
}
failed=$(find "$OUT/stations" -name failure.json | wc -l)
log "copula stage: $failed failed units"

guard "evaluate"
run evaluate "${CPU_ENV[@]}" "${APP[@]}" evaluate --config "$CONFIG" --out "$OUT" || exit 1
run paired_summary "${CPU_ENV[@]}" scripts/paper/application/paired_summary.py --run "$OUT" --out "$OUT/analysis" || exit 1
du -sh "$OUT" "$OUT/stations" > "$L/size.txt" 2>&1
log "all done"
