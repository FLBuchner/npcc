"""Shared paths, provenance and CSV helpers for experiment E3 (NPCC vs VDC).

Imported by every E3 script, in *both* interpreters:

- the VDC environment, ``/home/thibault/vdc/env/bin/python`` (Python 3.10,
  pyvinecopulib 0.7.5, VDC v0.2.0 installed editable), and
- the NPCC environment, ``/home/thibault/npcc/.venv`` (Python 3.12,
  pyvinecopulib 1.0, tabpfn 9.2), run as ``uv run --no-sync python ...``.

So this module stays Python 3.10-compatible and imports only the standard
library. It never imports either package under study: the versions of the
*other* environment are read by running that environment's interpreter in
isolated mode (``-I``) and asking ``importlib.metadata``.

Every output row of E3 carries the flat provenance from :func:`provenance`,
so a CSV can be read on its own: the VDC commit (the v0.2.0 tag commit and
the main-branch merge commit whose tree is identical), the checkpoint
sha256, the pyvinecopulib version of each environment, the npcc commit, the
tabpfn version, and -- in ``code_sha256`` -- the sha256 of the script that
wrote the row and of every E3 module it had loaded, which ``e3_collect.py``
checks against the files on disk, so a row produced by any other version of
the code is caught.

:func:`save_npz` writes an ``.npz`` byte for byte reproducibly (fixed member
timestamps and order), so regenerating the samples regenerates their sha256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

VDC_REPO = Path("/home/thibault/vdc/vine-denoising-copula")
VDC_PYTHON = Path("/home/thibault/vdc/env/bin/python")
NPCC_REPO = Path("/home/thibault/npcc")
NPCC_PYTHON = NPCC_REPO / ".venv" / "bin" / "python"
RESULTS = NPCC_REPO / "results" / "e3_vdc"
#: The directory of the E3 scripts; every module here is fingerprinted.
E3_DIR = Path(__file__).resolve().parent

CHECKPOINT = Path(
  "/home/thibault/vdc/weights/vdc-denoiser-m64-v1/vdc-denoiser-m64-v1.pt"
)
CHECKPOINT_SHA256 = (
  "8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f"
)
TABLE1_RUNNER = (
  VDC_REPO / "paper" / "neurips2026" / "runners" / "reproduce_table1_likelihood.py"
)
#: The v0.2.0 tag points at e09f3d3; the merge of the release into main is
#: d5aaa46. Both have tree db4fe77, so either names the code that ran.
VDC_TAG = "v0.2.0"
VDC_COMMIT = "e09f3d3"
VDC_MAIN_MERGE_COMMIT = "d5aaa46"

#: The log-density floor VDC's Table 1 runner applies to every density,
#: true and fitted alike: ``np.log(np.maximum(pdf, 1e-300))``.
LOG_FLOOR = 1e-300

_VERSION_PROBE = (
  "import importlib.metadata as m, json, platform\n"
  "out = {'python': platform.python_version()}\n"
  "for name in NAMES:\n"
  "    try:\n"
  "        out[name] = m.version(name)\n"
  "    except m.PackageNotFoundError:\n"
  "        out[name] = None\n"
  "print(json.dumps(out))\n"
)

VDC_ENV_PACKAGES = (
  "pyvinecopulib",
  "torch",
  "numpy",
  "scipy",
  "vine-denoising-copula",
)
NPCC_ENV_PACKAGES = (
  "pyvinecopulib",
  "tabpfn",
  "torch",
  "numpy",
  "scipy",
  "lightgbm",
  "xgboost-cpu",
  "catboost",
  "tabicl",
  "synthefy-nori",
)


def sha256_file(path: Path | str) -> str:
  """Return the hex sha256 of a file."""
  h = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
      h.update(block)
  return h.hexdigest()


def git_short(repo: Path) -> str:
  """Return ``git rev-parse --short HEAD`` of ``repo``."""
  return subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def git_dirty_paths(repo: Path, *paths: str) -> list[str]:
  """``git status --porcelain`` lines of ``repo`` restricted to ``paths``."""
  out = subprocess.run(
    ["git", "-C", str(repo), "status", "--porcelain", "--", *paths],
    check=True,
    capture_output=True,
    text=True,
  ).stdout
  return [line.strip() for line in out.splitlines() if line.strip()]


def git_src_dirty(repo: Path) -> bool:
  """Whether ``repo`` has uncommitted changes under ``src/`` or ``configs/``."""
  return bool(git_dirty_paths(repo, "src", "configs"))


def env_versions(python: Path, names: tuple[str, ...]) -> dict:
  """Package versions of another interpreter, without importing them here."""
  out = subprocess.run(
    [str(python), "-I", "-c", _VERSION_PROBE.replace("NAMES", repr(list(names)))],
    check=True,
    capture_output=True,
    text=True,
  ).stdout
  return json.loads(out)


def code_hashes(script: Path | str) -> dict[str, str]:
  """sha256 of ``script`` and of every E3 module this process has loaded.

  Called after a script's imports, so the map names exactly the E3 code that
  ran. ``e3_collect.py`` compares it with the files on disk.
  """
  files = {Path(script).resolve()}
  for module in list(sys.modules.values()):
    origin = getattr(module, "__file__", None)
    # Some extension modules report a bare relative name (torch's
    # `_classes.py`), which would resolve against the working directory.
    if not origin or not Path(origin).is_absolute():
      continue
    path = Path(origin).resolve()
    if path.parent == E3_DIR and path.exists():
      files.add(path)
  return {f.name: sha256_file(f) for f in sorted(files)}


def save_npz(path: Path, arrays: dict) -> None:
  """Write ``arrays`` as a deflated ``.npz``, byte for byte reproducibly.

  ``np.savez_compressed`` stamps the current time into every member, so the
  same arrays written twice hash differently; here the members carry a fixed
  timestamp and a sorted order, and the file's sha256 depends on the arrays
  alone (for a given zlib).
  """
  import numpy as np

  tmp = path.with_suffix(path.suffix + ".tmp")
  with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name in sorted(arrays):
      buf = io.BytesIO()
      np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
      info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = zipfile.ZIP_DEFLATED
      info.external_attr = 0o644 << 16
      zf.writestr(info, buf.getvalue())
  tmp.replace(path)


def provenance(script: Path | str, device: str) -> dict:
  """Flat provenance shared by every row a script writes.

  Parameters
  ----------
  script : path
      The script writing the rows; its sha256 is recorded, since the E3
      scripts are not committed and the npcc commit alone does not pin them.
  device : str
      The device the estimators of this script ran on.

  Returns
  -------
  dict
      Column name to value; every value is a scalar.
  """
  vdc_env = env_versions(VDC_PYTHON, VDC_ENV_PACKAGES)
  npcc_env = env_versions(NPCC_PYTHON, NPCC_ENV_PACKAGES)
  return {
    "vdc_tag": VDC_TAG,
    "vdc_commit": git_short(VDC_REPO),
    "vdc_main_merge_commit": VDC_MAIN_MERGE_COMMIT,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "pyvinecopulib_vdc_env": vdc_env["pyvinecopulib"],
    "pyvinecopulib_npcc_env": npcc_env["pyvinecopulib"],
    "torch_vdc_env": vdc_env["torch"],
    "torch_npcc_env": npcc_env["torch"],
    "numpy_vdc_env": vdc_env["numpy"],
    "numpy_npcc_env": npcc_env["numpy"],
    "tabpfn_version": npcc_env["tabpfn"],
    "lightgbm_version": npcc_env["lightgbm"],
    "xgboost_version": npcc_env["xgboost-cpu"],
    "catboost_version": npcc_env["catboost"],
    "tabicl_version": npcc_env["tabicl"],
    "nori_version": npcc_env["synthefy-nori"],
    "npcc_commit": git_short(NPCC_REPO),
    "npcc_src_dirty": git_src_dirty(NPCC_REPO),
    # The estimators live in src/npcc/core; a parallel session may hold
    # uncommitted work elsewhere in the tree, which this tells apart.
    "npcc_core_dirty": bool(git_dirty_paths(NPCC_REPO, "src/npcc/core")),
    "npcc_dirty_paths": ";".join(git_dirty_paths(NPCC_REPO, "src", "configs")),
    "script": Path(script).name,
    "script_sha256": sha256_file(script),
    "code_sha256": json.dumps(code_hashes(script), sort_keys=True),
    "device": device,
    "gpu": gpu_name() if device.startswith("cuda") else None,
    "hostname": platform.node(),
    "cpu_model": cpu_model(),
    "cpu_count": os.cpu_count(),
    "cpu_affinity": ",".join(str(c) for c in sorted(os.sched_getaffinity(0))),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "nice": os.nice(0),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  }


def cpu_model() -> str | None:
  """The ``model name`` line of ``/proc/cpuinfo``."""
  for line in Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("model name"):
      return line.split(":", 1)[1].strip()
  return None


def gpu_name() -> str | None:
  """The first GPU's name, from ``nvidia-smi``."""
  try:
    return subprocess.run(
      ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
      check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
  except (OSError, subprocess.CalledProcessError, IndexError):
    return None


def load_average() -> float:
  """One-minute load average: the machine is shared, so timings carry it."""
  return os.getloadavg()[0]


def write_csv(rows: list[dict], path: Path) -> None:
  """Write ``rows`` as a CSV whose header is the union of their keys."""
  if not rows:
    raise ValueError(f"rows must be non-empty to write {path}; got none")
  header: list[str] = []
  for row in rows:
    for key in row:
      if key not in header:
        header.append(key)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  tmp.replace(path)


def write_json(payload: dict, path: Path) -> None:
  """Write ``payload`` as indented JSON, atomically."""
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  tmp.replace(path)
