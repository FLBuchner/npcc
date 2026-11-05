#!/usr/bin/env python3
"""E3: run the pending GPU arms (TabPFN v2.5 / v3 / v3.5, TabICL, Nori).

Launch only once the GPU is free of E1:

    cd /home/thibault/npcc
    uv run --no-sync python scripts/paper/e3/e3_run_gpu_arms.py

It runs, in order and each in its own process, ``e3a_npcc_fit.py`` and
``e3b_npcc_fit.py`` with ``--estimators gpu --device cuda --gpu-is-free``,
then ``e3_collect.py``; logs go to ``results/e3_vdc/e3{a,b}_npcc_gpu.log``.
Both fit scripts resume from their per-fit files, so an interrupted run is
continued by running this again.

``PYTORCH_CUDA_ALLOC_CONF`` is removed from the children's environment: on
this machine's vGPU an ``expandable_segments`` setting makes every CUDA
allocation fail, since it lacks the virtual-memory API that mode needs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/thibault/npcc")
RESULTS = REPO / "results" / "e3_vdc"


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
  gpu = ("--estimators", "gpu", "--device", "cuda", "--gpu-is-free")
  run("e3a_npcc_fit.py", *gpu, log=RESULTS / "e3a_npcc_gpu.log")
  run("e3b_npcc_fit.py", *gpu, log=RESULTS / "e3b_npcc_gpu.log")
  run("e3_collect.py")


if __name__ == "__main__":
  main()
