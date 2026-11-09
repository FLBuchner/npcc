"""VDC as a pyvinecopulib 1.0 pair copula, served from VDC's own environment.

Imported in the NPCC environment (pyvinecopulib 1.0) by the E3 scripts.

**Why.** pyvinecopulib PR #217 integrated VDC into pyvinecopulib's torch
backend (``FitControlsTorchBicop(method="vdc")``, ``_fit_vdc.py``). That API
predates 1.0 and 1.0 has no VDC method, so this module adapts it rather than
copying it: the VDC estimate is the same, and the pair copula is 1.0's
grid-held one, :class:`pyvinecopulib.torch.TorchTllBicop`, built directly from
a grid. VDC, TLL and NPCC then run in the same vine machinery, so their
likelihoods and timings differ only through the pair-copula estimator.

**The adaptation** (as in PR #217): VDC's ``64 x 64`` density is cell-centered
on ``(i + 0.5) / 64``. It is replicate-padded to ``66 x 66`` on
``[0, 0.5/64, ..., 63.5/64, 1]`` and handed to ``TorchTllBicop`` with
``norm_maxiter=0``, which keeps the values exactly as given. VDC's Sinkhorn
projection already made the midpoint-rule margins uniform, and on the padded
grid the trapezoidal integral equals that midpoint sum at every cell center,
so the margins stay uniform and the h-functions agree with VDC's own
``(cumsum - density / 2) / m`` at the cell centers. The density is bilinear
between cell centers and constant beyond the outer ones -- what VDC's own
lookup (``HFuncLookup(use_spline=False).pdf``) does, so the two densities
should agree everywhere in the open square.

**Where VDC runs.** VDC's code stays in VDC's environment: :class:`VdcServer`
starts ``e3_vdc_server.py`` under ``/home/thibault/vdc/env/bin/python`` and
sends it each edge's pseudo-observations; the server returns the grid and the
seconds the estimate took inside it. The client's round-trip time minus that
is the inter-process overhead, reported separately.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from pyvinecopulib.torch import TorchTllBicop

import e3_common as common

HERE = Path(__file__).resolve().parent
HEADER = struct.Struct("<4I")
REPLY = struct.Struct("<2I2d")
#: Projection iterations of VDC's Table 1 runner and of its UCI vine runner.
DEFAULT_PROJECTION_ITERS = 30


class VdcServer:
  """A running ``e3_vdc_server.py``, in VDC's environment.

  Parameters
  ----------
  cpu_core : int, or None
      Pin the server to this core with ``taskset``; ``None`` leaves it free.
  device : str
      VDC's device, ``"cpu"`` here (the GPU is reserved).
  """

  def __init__(self, *, cpu_core: int | None = None, device: str = "cpu") -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cmd = [str(common.VDC_PYTHON), str(HERE / "e3_vdc_server.py"), "--device", device]
    if cpu_core is not None:
      cmd = ["taskset", "-c", str(cpu_core), *cmd]
    self.proc = subprocess.Popen(
      cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=str(HERE))
    self.calls: list[dict[str, float]] = []
    self.info = self._info()

  def _read(self, size: int) -> bytes:
    assert self.proc.stdout is not None
    data = self.proc.stdout.read(size)
    if data is None or len(data) != size:
      raise RuntimeError(f"VDC server must answer {size} bytes; got {0 if data is None else len(data)}")
    return data

  def _info(self) -> dict[str, Any]:
    assert self.proc.stdin is not None
    self.proc.stdin.write(HEADER.pack(2, 0, 0, 0))
    self.proc.stdin.flush()
    (length,) = struct.unpack("<I", self._read(4))
    return json.loads(self._read(length))

  def estimate(
    self, u: np.ndarray, projection_iters: int = DEFAULT_PROJECTION_ITERS
  ) -> tuple[np.ndarray, dict[str, float]]:
    """VDC's normalized ``64 x 64`` grid for ``u``, and its timings."""
    u = np.ascontiguousarray(u, dtype="<f8")
    if u.ndim != 2 or u.shape[1] != 2:
      raise ValueError(f"u must have shape (n, 2); got {u.shape}")
    assert self.proc.stdin is not None
    t0 = time.perf_counter()
    self.proc.stdin.write(HEADER.pack(1, u.shape[0], projection_iters, 0) + u.tobytes())
    self.proc.stdin.flush()
    status, m, server_s, projection_s = REPLY.unpack(self._read(REPLY.size))
    grid = np.frombuffer(self._read(m * m * 8), dtype="<f8").reshape(m, m).copy()
    roundtrip_s = time.perf_counter() - t0
    if status != 0:
      raise RuntimeError(f"VDC server must return status 0; got {status}")
    timing = {"server_s": server_s, "projection_s": projection_s,
              "roundtrip_s": roundtrip_s, "ipc_s": roundtrip_s - server_s}
    self.calls.append(timing)
    return grid, timing

  def close(self) -> None:
    """Ask the server to quit and wait for it."""
    if self.proc.poll() is None:
      assert self.proc.stdin is not None
      try:
        self.proc.stdin.write(HEADER.pack(0, 0, 0, 0))
        self.proc.stdin.flush()
        self.proc.stdin.close()
      except BrokenPipeError:
        pass
      self.proc.wait(timeout=60)

  def __enter__(self) -> VdcServer:
    return self

  def __exit__(self, *exc: object) -> None:
    self.close()


def replicate_pad(density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Pad a cell-centered ``(m, m)`` density to ``(m + 2, m + 2)`` on ``[0, 1]``.

  The grid points are ``[0, 0.5/m, 1.5/m, ..., (m - 0.5)/m, 1]`` and the new
  first and last rows and columns repeat their neighbors (PR #217's
  ``_replicate_pad``).
  """
  if density.ndim != 2 or density.shape[0] != density.shape[1]:
    raise ValueError(f"density must be square and 2-D; got {tuple(density.shape)}")
  m = density.shape[0]
  centers = (torch.arange(m, dtype=density.dtype, device=density.device) + 0.5) / m
  points = torch.cat([centers.new_zeros(1), centers, centers.new_ones(1)])
  padded = torch.nn.functional.pad(density[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
  return points, padded


class VdcBicop(TorchTllBicop):
  """VDC's pair-copula estimate as a pyvinecopulib 1.0 ``TorchTllBicop``.

  ``from_data`` sends the pseudo-observations to :attr:`server`, pads the
  returned grid and builds the grid-held pair copula with
  ``norm_maxiter=0``; everything else -- ``pdf``, ``cdf``, the h-functions and
  their inverses, sampling -- is ``TorchTllBicop``'s. Set :attr:`server`
  (and, if wanted, :attr:`projection_iters`) before fitting.
  """

  server: ClassVar[VdcServer | None] = None
  projection_iters: ClassVar[int] = DEFAULT_PROJECTION_ITERS

  @classmethod
  def from_data(
    cls,
    u: np.ndarray | torch.Tensor,
    /,
    controls: object = None,
    *,
    var_types: list[str] | None = None,
    x: torch.Tensor | None = None,
    cache_integrals: bool | None = None,
    device: torch.types.Device = None,
    dtype: torch.dtype | None = None,
  ) -> VdcBicop:
    """Estimate the pair copula with VDC and wrap it as a 1.0 grid pair."""
    del controls
    if x is not None:
      raise ValueError("x must be None: VDC's estimate is unconditional")
    if var_types is not None and "d" in var_types:
      raise ValueError(f"var_types must be continuous for VDC; got {var_types}")
    if cls.server is None:
      raise RuntimeError("VdcBicop.server must be set to a running VdcServer before fitting")
    u_np = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else np.asarray(u)
    # The Table 1 runner's sample clip; a no-op on its samples, and it keeps a
    # vine's pseudo-observations off the boundary of VDC's histogram.
    u_np = np.clip(np.asarray(u_np, dtype=np.float64)[:, :2], 1e-10, 1 - 1e-10)
    grid, timing = cls.server.estimate(u_np, cls.projection_iters)
    dtype = torch.float64 if dtype is None else dtype
    points, values = replicate_pad(torch.as_tensor(grid, dtype=dtype, device=device))
    pair = cls(grid_points=points, values=values,
               cache_integrals=True if cache_integrals is None else cache_integrals,
               norm_maxiter=0, is_linear=False, device=device, dtype=dtype)
    pair.vdc_timing = timing
    pair.vdc_grid = grid
    return pair
