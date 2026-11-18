#!/usr/bin/env bash
# Baobab, on the job's GPU node: a worktree at WANT on the node's own disk,
# then launch_full.sh into BeeGFS scratch, with the GPU's use logged.
#   WANT=<sha> KIND=smoke|full [RESUME=1] [OUT=..] [GPU_SHARDS=..] [SHARDS=..] \
#     bash start.sh
# A worktree per commit, so a running launch never sees its code change; the
# env is setup_node.sh's, and PYTHONPATH points its npcc at the worktree.
set -euo pipefail
: "${WANT:?the commit to run}" "${KIND:?smoke or full}"
B=${B:-$HOME/scratch}
A=$B/npcc-app
N=${N:-/scratch/$USER-npcc}
cd "$N/npcc"
git fetch -q "$A/npcc.bundle" 'refs/heads/*:refs/remotes/bundle/*'
W=$N/wt-${WANT:0:7}
[ -d "$W" ] || git worktree add -q --detach "$W" "$WANT"
case "$KIND" in
  smoke) CONFIG=configs/application_full_smoke.toml; OUT=${OUT:-$B/npcc-runs/application_full_smoke_$(date +%Y-%m-%d)};;
  full) CONFIG=configs/application_full.toml; OUT=${OUT:-$B/npcc-runs/application_full_$(date +%Y-%m-%d)};;
  *) echo "KIND must be smoke or full; got $KIND"; exit 1;;
esac
mkdir -p "$B/npcc-runs"
TAG=$(basename "$OUT")
nohup bash -c "while :; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> $B/npcc-runs/$TAG.gpu.csv; sleep 15; done" > /dev/null 2>&1 < /dev/null &
echo $! > "$B/npcc-runs/$TAG.gpu.pid"
WANT=$WANT W=$W OUT=$OUT PY=$N/npcc/.venv/bin/python CONFIG=$CONFIG \
  SETUP=$W/scripts/paper/application/baobab/env.sh RESUME=${RESUME:-0} \
  GPU_SHARDS=${GPU_SHARDS:-1} SHARDS=${SHARDS:-2} MIN_FREE_GB=${MIN_FREE_GB:-20} \
  setsid -f nohup bash "$W/scripts/paper/application/launch_full.sh" \
  > "$B/npcc-runs/$TAG.launcher.out" 2>&1 < /dev/null
echo "started $KIND at $WANT into $OUT on $(hostname)"
