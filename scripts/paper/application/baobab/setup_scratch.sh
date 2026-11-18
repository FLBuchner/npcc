#!/usr/bin/env bash
# Baobab, login node: the code from a git bundle on BeeGFS scratch, where the
# run folders and the data live. The repository is private and the cluster
# holds no GitHub credentials, so the code travels as a bundle:
#   git bundle create npcc.bundle <branch>; scp npcc.bundle baobab:scratch/npcc-app/
# The data go to $B/npcc-data/meteoswiss_zanetta2023 (x.nc, y.nc, checked
# against their SHA-256), linked from ~/npcc-data, the configs' data dir.
set -euo pipefail
B=${B:-$HOME/scratch}
A=$B/npcc-app
[ -d "$A/npcc" ] || git clone -q "$A/npcc.bundle" "$A/npcc"
cd "$A/npcc"
git fetch -q "$A/npcc.bundle" 'refs/heads/*:refs/remotes/bundle/*'
[ -e "$HOME/npcc-data" ] || ln -s "$B/npcc-data" "$HOME/npcc-data"
echo "bundle heads: $(git for-each-ref --format='%(refname:short) %(objectname:short)' refs/remotes/bundle | tr '\n' ' ')"
