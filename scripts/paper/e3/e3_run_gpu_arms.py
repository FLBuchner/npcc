#!/usr/bin/env python3
"""E3: run the pending GPU arms (TabPFN v2.5 / v3 / v3.5, TabICL, Nori).

Launch only once the GPU is free of E1, naming the main-text TabPFN version:

    cd /home/thibault/npcc
    uv run --no-sync python scripts/paper/e3/e3_run_gpu_arms.py --main-tabpfn v3.5

It runs, in order and each in its own process:

1. E3a at ``n_train = 2000`` for every GPU estimator, and at ``n_train =
   10000`` for the main-text TabPFN version only (the paper lead's scope);
2. E3b for every GPU estimator;
3. E3c for every GPU estimator, one vine fit per arm and cell;
4. ``e3_collect.py``.

Logs go to ``results/e3_vdc/e3{a,b,c}_npcc_gpu.log``. Every fit script
resumes from its per-fit files, so an interrupted run continues when this is
run again; ``--stages`` runs a subset.

``PYTORCH_CUDA_ALLOC_CONF`` is removed from the children's environment: on
this machine's vGPU an ``expandable_segments`` setting makes every CUDA
allocation fail, since it lacks the virtual-memory API that mode needs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/thibault/npcc")
RESULTS = REPO / "results" / "e3_vdc"
sys.path.insert(0, str(HERE))


def run(script: str, *args: str, log: Path | None = None) -> None:
  """Run one E3 script with ``uv run --no-sync python`` from the repo root."""
  env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
  cmd = ["uv", "run", "--no-sync", "python", str(HERE / script), *args]
  print("$", " ".join(cmd), flush=True)
  with (log.open("a") if log else open(os.devnull, "w")) as sink:
    proc = subprocess.run(cmd, cwd=REPO, env=env,
                          stdout=sink if log else None, stderr=subprocess.STDOUT if log else None)
  if proc.returncode != 0:
    sys.exit(f"{script} must exit 0; got {proc.returncode} (see {log})")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--main-tabpfn", required=True, choices=["v2.5", "v3", "v3.5"],
                 help="the main-text TabPFN version, the only GPU arm at n_train = 10000")
  p.add_argument("--stages", nargs="+", default=["e3a", "e3b", "e3c"],
                 choices=["e3a", "e3b", "e3c"])
  a = p.parse_args()

  import e3_npcc

  gpu = ("--device", "cuda", "--gpu-is-free")
  main_tabpfn = f"tabpfn-quantiles-{a.main_tabpfn}-logit"
  if "e3a" in a.stages:
    log = RESULTS / "e3a_npcc_gpu.log"
    run("e3a_npcc_fit.py", "--estimators", "gpu", "--n-train", "2000", *gpu, log=log)
    run("e3a_npcc_fit.py", "--estimators", main_tabpfn, "--n-train", "10000", *gpu, log=log)
  if "e3b" in a.stages:
    run("e3b_npcc_fit.py", "--estimators", "gpu", *gpu, log=RESULTS / "e3b_npcc_gpu.log")
  if "e3c" in a.stages:
    arms = [f"npcc-{label}" for label in e3_npcc.GPU_ESTIMATORS]
    run("e3c_npcc_fit.py", "--arms", *arms, *gpu, log=RESULTS / "e3c_npcc_gpu.log")
  run("e3_collect.py")


if __name__ == "__main__":
  main()
