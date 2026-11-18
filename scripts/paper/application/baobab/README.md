# The full grid on Baobab

How `configs/application_full.toml` runs on UNIGE's Baobab cluster, in an
interactive GPU allocation (one A100 80 GB, 4 CPUs, 32 GB, 5 days, on
`private-ruch-gpu`), started from a tmux session on the login node so it
survives disconnects.

1. On this machine: `git bundle create npcc.bundle <branch>`, then copy it,
   and the data with their SHA-256, to `~/scratch/npcc-app/` and
   `~/scratch/npcc-data/meteoswiss_zanetta2023/`; copy the `uv` binary to
   `~/scratch/npcc-app/bin/`.
2. Login node: `bash setup_scratch.sh` (the code, and `~/npcc-data`).
3. GPU node (`ssh <node>` lands in the allocation): `bash setup_node.sh`,
   the env and boostCopula's R library on the node's own disk,
   `/scratch/$USER-npcc`. BeeGFS scratch is too slow to import torch from.
   The node's disk may not outlive the allocation; a new allocation reruns
   this script.
4. GPU node: `WANT=<sha> KIND=smoke bash start.sh`, the station-0 smoke;
   then `KIND=full`. A second allocation continues a stopped run with
   `RESUME=1 OUT=<run folder>`.

The run folders, `~/scratch/npcc-runs/`, and the stored draws stay on BeeGFS
scratch. TabPFN's and TabICL's weights are copied into `~/.cache` there,
without TabPFN's auth token, which loading the weights does not need.
