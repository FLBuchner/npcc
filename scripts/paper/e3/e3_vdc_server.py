#!/usr/bin/env python3
"""VDC density server: VDC's pair-density estimate, served to another process.

Runs in the VDC environment (Python 3.10, VDC v0.2.0, pyvinecopulib 0.7.5)
and is started by ``e3_vdc_pair.VdcServer`` from the NPCC environment, which
keeps VDC's code in its own interpreter while the vine machinery runs in
pyvinecopulib 1.0. Not meant to be run by hand.

**What it computes.** For each request, an ``(n, 2)`` float64 sample of
pseudo-observations, it returns the ``64 x 64`` cell-centred density grid of
the released ``vdc-denoiser-m64-v1``, exactly as VDC's Table 1 runner computes
it: ``estimate_pair_density_from_samples(bundle, u, m=64,
projection_iters=k)`` under ``torch.no_grad()``, then ``grid /= grid.mean()``.
``k`` is sent with each request (the runner's 30 by default). The model is
loaded as the runner loads it (``load_checkpoint_bundle``, then a strict
``load_state_dict``) with the runner's numerical settings: one torch thread,
matrix TF32 off, convolution TF32 on, cuDNN benchmark off. One untimed
warm-up estimate is made at start-up.

**Timing.** ``server_s`` is the estimate itself (histogram, denoiser forward
pass, Sinkhorn projection, normalization) inside this process, and
``projection_s`` the projection alone, timed by wrapping
``vdc.models.projection.copula_project`` (which VDC's estimator imports at
call time). The client measures the round trip; their difference is the
inter-process overhead.

**Protocol** (little-endian, over stdin / the original stdout; anything the
libraries print goes to stderr):

- request: ``uint32 op, uint32 n, uint32 projection_iters, uint32 0``, then
  ``n * 2`` float64 values (row-major) when ``op == 1``; ``op == 0`` quits,
  ``op == 2`` asks for the server's provenance as JSON;
- reply to ``op == 1``: ``uint32 status, uint32 m, float64 server_s, float64
  projection_s``, then ``m * m`` float64 values (row ``i`` is ``u``'s cell
  ``i``, as VDC's lookup reads them);
- reply to ``op == 2``: ``uint32 length``, then that many bytes of JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402

HEADER = struct.Struct("<4I")
REPLY = struct.Struct("<2I2d")


def read_exact(stream, size: int) -> bytes:  # noqa: ANN001 - a binary stream
  """Read exactly ``size`` bytes, or raise ``EOFError``."""
  chunks, remaining = [], size
  while remaining:
    chunk = stream.read(remaining)
    if not chunk:
      raise EOFError("client closed the pipe")
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  a = p.parse_args()

  # Keep the protocol on a private copy of stdout; library output goes to stderr.
  proto_out = os.fdopen(os.dup(1), "wb")
  os.dup2(2, 1)
  sys.stdout = sys.stderr
  proto_in = sys.stdin.buffer

  import torch

  import vdc
  import vdc.models.projection as projection
  from vdc.pretrained import estimate_pair_density_from_samples, load_checkpoint_bundle

  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from e3a_vdc_side import ProjectionTimer

  assert Path(vdc.__file__).resolve().is_relative_to(common.VDC_REPO)
  checkpoint_sha = common.sha256_file(a.checkpoint)
  assert checkpoint_sha == common.CHECKPOINT_SHA256
  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = False
  t0 = time.perf_counter()
  bundle = load_checkpoint_bundle(a.checkpoint, device=a.device)
  ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
  bundle.model.load_state_dict(ck["model_state_dict"], strict=True)
  del ck
  load_s = time.perf_counter() - t0
  timer = ProjectionTimer(projection)
  with torch.no_grad():
    estimate_pair_density_from_samples(
      bundle, np.random.default_rng(0).uniform(size=(2000, 2)), m=64, projection_iters=30)
  timer.take()

  info = {
    "server": Path(__file__).name, "server_sha256": common.sha256_file(__file__),
    "code_sha256": common.code_hashes(__file__),
    "checkpoint_sha256": checkpoint_sha, "vdc_commit": common.git_short(common.VDC_REPO),
    "vdc_module": str(Path(vdc.__file__).resolve()),
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "torch": torch.__version__},
    "device": a.device, "torch_threads": torch.get_num_threads(), "model_load_s": load_s,
    "settings": {"m": 64, "normalization": "grid /= grid.mean()", "dtype": "float32 model",
                 "matmul_tf32": False, "cudnn_tf32": True, "cudnn_benchmark": False},
    "cpu_affinity": ",".join(str(c) for c in sorted(os.sched_getaffinity(0))),
  }
  while True:
    try:
      op, n, iters, _ = HEADER.unpack(read_exact(proto_in, HEADER.size))
    except EOFError:
      return
    if op == 0:
      return
    if op == 2:
      payload = json.dumps(info).encode()
      proto_out.write(struct.pack("<I", len(payload)) + payload)
      proto_out.flush()
      continue
    u = np.frombuffer(read_exact(proto_in, n * 2 * 8), dtype="<f8").reshape(n, 2)
    timer.take()
    t0 = time.perf_counter()
    with torch.no_grad():
      grid = np.asarray(estimate_pair_density_from_samples(
        bundle, u, m=64, projection_iters=int(iters)), dtype=float)
    grid /= grid.mean()
    server_s = time.perf_counter() - t0
    projection_s = timer.take()
    m = grid.shape[0]
    proto_out.write(REPLY.pack(0, m, server_s, projection_s)
                    + np.ascontiguousarray(grid, dtype="<f8").tobytes())
    proto_out.flush()


if __name__ == "__main__":
  main()
