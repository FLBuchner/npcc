"""E11 on E3d: the chain mixture against the vines on the same samples.

Reads the chain's cells (``scripts/paper/e3/e3d_chain.py``) from ``--chain``
and scores each against every reference it has, paired by cell on the test
points where both are finite, as ``scripts/paper/e10/compare_e3d.py`` scores
E10's mixture (whose ``compare`` and ``summarize`` this uses): excess NLL =
mean of ``log c_true - log max(c_hat, 1e-300)``. The references:

- ``orders_mean`` and ``best_order``: the chain's own orders, where the
  mixture and every order are finite; the best order is the lowest there,
  an oracle choice no user has;
- ``e3d:<arm>``, from E3d's folder (``--e3d``): the same estimator's
  non-simplified vine on TLL's selected structure
  (``npcc-<estimator>-tll-ns``), and VDC's pair copulas on the true
  structure (``vdc_true``);
- ``e10:<arm>``, from an E10-on-E3d folder (``--e10``): the same
  estimator's mixture over five structures (``npcc-<estimator>``), and
  ``e10:<arm>:m1``, its member 1, which refits ``tll-ns`` there;
- ``e3e:<arm>``, from an E3e folder (``--e3e``): the same estimator's
  ``tll-s``, ``true-ns`` and ``true-s``, and ``tll-ns`` where that folder
  holds it.

A reference a folder lacks for a cell is left out of that cell. At d = 10
two machines' TabPFN vines differ by up to 4e-3 in excess NLL per seed, so
the references to read there are those fitted on the chain's machine: pass
that machine's E10 and E3e folders, whose references are named apart from
E3d's.

Writes ``compare_cells.csv`` (one row per cell and reference) and
``compare.csv`` (per design, d and reference: the mean difference chain -
reference over seeds, its SE, and in how many seeds the chain is lower), in
``compare_e3d.py``'s columns, ``mixture`` being the chain's, into ``--out``.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e11/compare_e3d_chain.py \\
        --chain results/e11_e3d_<TAG> --e10 results/e10_e3d_2026-11-11 \\
        --e3e results/e3e_simplification_2x2 --dims 5 \\
        --out results/e11_e3d_<TAG>/analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e10"))
from compare_e3d import compare, excess, summarize

#: The main-text TabPFN, E1's label.
ESTIMATOR = "tabpfn-quantiles-v3-probit"
E3D = (
  Path(os.environ.get("E3_NPCC_REPO", "/home/thibault/npcc"))
  / "results"
  / "e3d_unknown_structure"
)
E3E_VARIANTS = ("tll-ns", "tll-s", "true-ns", "true-s")


def references(
  key: str, label: str, *, e3d: Path, e10: Path, e3e: Path
) -> dict[str, np.ndarray]:
  """Every reference the three folders hold for the cell ``key``."""
  npcc = f"npcc-{label}"
  files = [
    (f"e3d:{npcc}-tll-ns", e3d / "cells" / f"{npcc}-tll-ns"),
    ("e3d:vdc_true", e3d / "cells" / "vdc_true"),
    (f"e10:{npcc}", e10 / "cells" / npcc),
    *((f"e3e:{npcc}-{v}", e3e / "cells" / f"{npcc}-{v}") for v in E3E_VARIANTS),
  ]
  refs = {
    name: np.load(folder / f"{key}.npy")
    for name, folder in files
    if (folder / f"{key}.npy").is_file()
  }
  members = e10 / "cells" / npcc / f"{key}_members.npy"
  if members.is_file():
    refs[f"e10:{npcc}:m1"] = np.load(members)[0]
  return refs


def cell_rows(
  key: str,
  folder: Path,
  logtrue: np.ndarray,
  refs: dict[str, np.ndarray],
) -> list[dict[str, object]]:
  """One cell's rows: the chain against its orders and every reference."""
  design, dim, seed = key.split("_")
  base: dict[str, object] = {
    "design": design,
    "d": int(dim.removeprefix("d")),
    "seed": int(seed.removeprefix("s")),
  }
  mixture = np.load(folder / f"{key}.npy").astype(np.float64)
  orders = np.load(folder / f"{key}_members.npy").astype(np.float64)
  rows = compare(
    base, mixture, orders, refs, logtrue, members_name="orders_mean"
  )
  finite = np.isfinite(mixture) & np.isfinite(orders).all(axis=0)
  scores = [excess(logtrue, o, finite) for o in orders]
  best = int(np.argmin(scores))
  rows.insert(
    1,
    {
      **base,
      "reference": "best_order",
      "mixture": excess(logtrue, mixture, finite),
      "ref": scores[best],
      "n_common": int(finite.sum()),
      "best_order": best + 1,
    },
  )
  return rows


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--chain", type=Path, required=True)
  p.add_argument("--estimator", default=ESTIMATOR)
  p.add_argument("--e3d", type=Path, default=E3D)
  p.add_argument("--e10", type=Path, required=True)
  p.add_argument("--e3e", type=Path, required=True)
  p.add_argument("--designs", nargs="+", default=["decay", "strong"])
  p.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  with np.load(a.e3d / "samples.npz") as f:
    samples = {k: f[k] for k in f.files if k.endswith("_logtrue")}
  folder = a.chain / "cells" / f"chain-{a.estimator}"
  rows: list[dict[str, object]] = []
  for path in sorted(folder.glob("*_members.npy")):
    key = path.name.removesuffix("_members.npy")
    design, dim, _ = key.split("_")
    if design not in a.designs or int(dim.removeprefix("d")) not in a.dims:
      continue
    refs = references(key, a.estimator, e3d=a.e3d, e10=a.e10, e3e=a.e3e)
    logtrue = samples[f"{key}_logtrue"].astype(np.float64)
    rows += cell_rows(key, folder, logtrue, refs)
  if not rows:
    raise SystemExit(f"{folder} must hold finished chain cells; it has none")
  cells, summary = summarize(rows)
  a.out.mkdir(parents=True, exist_ok=True)
  cells.to_csv(a.out / "compare_cells.csv", index=False)
  summary.to_csv(a.out / "compare.csv", index=False)
  with pd.option_context("display.width", 200):
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
  main()
