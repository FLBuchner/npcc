"""E12: the total mass of E3d's, E3e's and E10's fitted vines, per cell.

Every vine of these studies is scored at the 20,000 test points of an E3d
cell, which are drawn from the cell's true vine. ``samples.npz`` in E3d's
folder holds the true log-density there (``<key>_logtrue``), and each arm's
``cells/<arm>/<key>.npy`` the fitted one, so the mass of every fitted vine
follows by importance sampling under the truth (``mass_common.stats``). Test
points where the fitted log-density is not finite are left out and counted.

**Inputs.** E3d's folder (``samples.npz`` and ``cells/``), and the
``cells/`` of E3e and E10 on E3d, each run here and on gpu053. A study whose
folder has no ``cells/`` is skipped. Every ``.npy`` present is read, so a
study still being written is covered as far as it has gone; rerun on the
finished folders to cover the rest.

**Outputs**, in ``--out``:

- ``mass_e3_cells.csv``: one row per study, arm and cell, with
  ``mass_common.stats``, ``n_nonfinite``, and from the cell's JSON row its
  ``excess_nll``, ``test_loglik`` and ``true_loglik`` (``json_*``) and
  ``check_true_ll``, the mean of the stored true log-density less the row's
  ``true_loglik`` (0 when the row and ``samples.npz`` agree);
- ``mass_e10_members.csv``: the same per member of every E10 mixture, from
  its ``_members.npy``.

``study`` names the folder: ``E3d``, ``E3e``, ``E3e_gpu053``, ``E10`` and
``E10_gpu053``. The two E10 folders hold the same competitor cells, so a
table that pools them must take each arm from one.

**Usage** (from the repository root, on the CPU, in seconds)::

    uv run --no-sync python scripts/paper/e12/mass_e3.py --out results/e12_<TAG>

``summarize.py``, ``final_tables.py``, ``paired.py`` and ``vdc_pairs.py``
read ``mass_e3_cells.csv`` from there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mass_common import RESULTS, stats


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument(
    "--e3d",
    type=Path,
    default=RESULTS / "e3d_unknown_structure",
    help="E3d's folder; its samples.npz serves every study",
  )
  p.add_argument("--e3e", type=Path, default=RESULTS / "e3e_simplification_2x2")
  p.add_argument(
    "--e3e-gpu053",
    type=Path,
    default=RESULTS / "e3e_simplification_2x2_gpu053",
  )
  p.add_argument("--e10", type=Path, default=RESULTS / "e10_e3d_2026-11-11")
  p.add_argument(
    "--e10-gpu053", type=Path, default=RESULTS / "e10_e3d_gpu053_2026-11-11"
  )
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()
  folders = {
    "E3d": a.e3d,
    "E3e": a.e3e,
    "E3e_gpu053": a.e3e_gpu053,
    "E10": a.e10,
    "E10_gpu053": a.e10_gpu053,
  }
  a.out.mkdir(parents=True, exist_ok=True)

  with np.load(a.e3d / "samples.npz") as f:
    logtrue = {
      k.removesuffix("_logtrue"): f[k].astype(np.float64)
      for k in f.files
      if k.endswith("_logtrue")
    }
  rows, member_rows = [], []
  for study, folder in folders.items():
    cells = folder / "cells"
    if not cells.is_dir():
      print(study, "skipped: no", cells, flush=True)
      continue
    for arm_dir in sorted(q for q in cells.iterdir() if q.is_dir()):
      arm = arm_dir.name
      for npy in sorted(arm_dir.glob("*.npy")):
        key = npy.stem
        if key.endswith("_members"):
          continue
        design, dd, ss = key.split("_")
        lt = logtrue[key]
        lp = np.load(npy).astype(np.float64)
        if lp.shape != lt.shape:
          raise SystemExit(
            f"{npy} must hold one value per test point, {lt.shape}; got "
            f"{lp.shape}"
          )
        fin = np.isfinite(lp)
        lr = lp[fin] - lt[fin]
        row = {
          "study": study,
          "arm": arm,
          "design": design,
          "d": int(dd[1:]),
          "seed": int(ss[1:]),
          "n_nonfinite": int((~fin).sum()),
          **stats(lr),
        }
        js = npy.with_suffix(".json")
        if js.exists():
          j = json.loads(js.read_text())
          row["json_excess_nll"] = j.get("excess_nll")
          row["json_test_loglik"] = j.get("test_loglik")
          row["json_true_loglik"] = j.get("true_loglik")
          row["check_true_ll"] = float(lt.mean()) - j.get("true_loglik", np.nan)
        rows.append(row)
        mem = npy.with_name(f"{key}_members.npy")
        if mem.exists():
          members = np.load(mem).astype(np.float64)
          for jj in range(members.shape[0]):
            fm = np.isfinite(members[jj])
            member_rows.append(
              {
                "study": study,
                "arm": arm,
                "design": design,
                "d": int(dd[1:]),
                "seed": int(ss[1:]),
                "member": jj + 1,
                "n_nonfinite": int((~fm).sum()),
                **stats(members[jj][fm] - lt[fm]),
              }
            )
      print(study, arm, "done", flush=True)
  df = pd.DataFrame(rows)
  df.to_csv(a.out / "mass_e3_cells.csv", index=False)
  pd.DataFrame(member_rows).to_csv(a.out / "mass_e10_members.csv", index=False)
  print(df.shape, len(member_rows))


if __name__ == "__main__":
  main()
