#!/usr/bin/env bash
# Baobab, on the GPU node: a worktree at WANT on the node's own disk, then
# launch.sh into a run folder on BeeGFS scratch, with the GPU's use logged.
#   WANT=<sha> OUT=<run folder> [STATIONS=0] [RESUME=1] [GPU_SHARDS=2] \
#     [SHARDS=2] bash start.sh
# STATIONS=0 is the station-0 smoke, RESUME=1 continues a stopped run in a
# new allocation. A worktree per commit, so a running launch never sees its
# code change; the env is setup.sh's, and PYTHONPATH points its npcc at the
# worktree.
set -euo pipefail
: "${WANT:?the commit to run}" "${OUT:?the run folder}"
B=${B:-$HOME/scratch}
A=$B/npcc-app
N=${N:-/scratch/$USER-npcc}
cd "$N/npcc"
git fetch -q "$A/npcc.bundle" 'refs/heads/*:refs/remotes/bundle/*'
W=$N/wt-${WANT:0:7}
[ -d "$W" ] || git worktree add -q --detach "$W" "$WANT"
mkdir -p "$(dirname "$OUT")"
nohup bash -c "while :; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> $OUT.gpu.csv; sleep 15; done" > /dev/null 2>&1 < /dev/null &
echo $! > "$OUT.gpu.pid"
WANT=$WANT W=$W OUT=$OUT PY=$N/npcc/.venv/bin/python STATIONS=${STATIONS:-} \
  SETUP=$W/scripts/paper/application/baobab/env.sh RESUME=${RESUME:-0} \
  GPU_SHARDS=${GPU_SHARDS:-2} SHARDS=${SHARDS:-2} MIN_FREE_GB=${MIN_FREE_GB:-20} \
  setsid -f nohup bash "$W/scripts/paper/application/launch.sh" \
  > "$OUT.launcher.out" 2>&1 < /dev/null
echo "started $WANT into $OUT on $(hostname)"
