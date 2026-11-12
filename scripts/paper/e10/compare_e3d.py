"""E10 on E3d: an NPCC mixture against every reference on the same samples.

Reads the NPCC arm's mixture cells from E10's folder
(``scripts/paper/e3/e3d_mixture.py``), the competitors' mixtures beside them,
and E3d's own rows (``results/e3d_unknown_structure``), and scores each pair
on the test points where both are finite, as ``e3_collect.py`` does: excess
NLL = mean of ``log c_true - log max(c_hat, 1e-300)``. The references:

- ``m1`` and ``members_mean``: the mixture's first member (the TLL-selected
  structure) and its members' mean, from its own ``_members.npy``;
- ``mixture:<m>``: each competitor's mixture on the same structures;
- ``e3d:<arm>``: E3d's rows, each competitor on its own selected structure
  (``tll``, ``parametric``, and VDC's own pipeline ``vdc``), VDC on the true
  structure (``vdc_true``), and the same NPCC estimator's non-simplified fit
  on the selected structure (``npcc-<est>-tll-ns``, which m1 refits).

Writes ``compare_cells.csv`` (one row per cell and reference) and
``compare.csv`` (per design, d and reference: the mean difference
mixture - reference over seeds, its SE, and in how many seeds the mixture is
lower) into ``--out``.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e10/compare_e3d.py \\
        --e10 results/e10_e3d_<TAG> --dims 5 --out results/e10_e3d_<TAG>/analysis
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR = math.log(1e-300)
COMPETITORS = ("tll", "parametric", "vdc_shared")
E3D_ARMS = ("tll", "parametric", "vdc", "vdc_true")


def excess(logtrue: np.ndarray, lp: np.ndarray, mask: np.ndarray) -> float:
  """Excess NLL over ``mask``, with the density floored as E3d floors it."""
  return float(np.mean((logtrue - np.maximum(lp, FLOOR))[mask]))


def compare(
  base: dict[str, object],
  mixture: np.ndarray,
  members: np.ndarray,
  refs: dict[str, np.ndarray],
  logtrue: np.ndarray,
  *,
  members_name: str = "members_mean",
) -> list[dict[str, object]]:
  """One row per reference: a mixture and the reference on shared points.

  The members' mean is scored where the mixture and every member are
  finite; each reference where the mixture and that reference are.
  E11's comparison (``scripts/paper/e11/compare_e3d_chain.py``) reads its
  chain mixture against its references through this too.
  """
  finite = np.isfinite(mixture) & np.isfinite(members).all(axis=0)
  rows: list[dict[str, object]] = [
    {
      **base,
      "reference": members_name,
      "mixture": excess(logtrue, mixture, finite),
      "ref": float(np.mean([excess(logtrue, m, finite) for m in members])),
      "n_common": int(finite.sum()),
    }
  ]
  for name, lp in refs.items():
    lp = np.asarray(lp, dtype=np.float64)
    both = np.isfinite(mixture) & np.isfinite(lp)
    rows.append(
      {
        **base,
        "reference": name,
        "mixture": excess(logtrue, mixture, both),
        "ref": excess(logtrue, lp, both),
        "n_common": int(both.sum()),
      }
    )
  return rows


def summarize(
  rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """The per-cell rows with their differences, and the summary over seeds.

  Per design, d and reference: the mean difference mixture - reference over
  seeds, its SE, and in how many seeds the mixture is lower.
  """
  cells = pd.DataFrame(rows)
  cells["diff"] = cells["mixture"] - cells["ref"]
  summary = (
    cells.groupby(["design", "d", "reference"])
    .agg(
      seeds=("diff", "size"),
      mixture=("mixture", "mean"),
      ref=("ref", "mean"),
      diff_mean=("diff", "mean"),
      diff_sd=("diff", "std"),
      mixture_lower=("diff", lambda s: int((s < 0).sum())),
    )
    .reset_index()
  )
  summary["diff_se"] = summary["diff_sd"] / np.sqrt(summary["seeds"])
  return cells, summary.drop(columns="diff_sd")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--e10", type=Path, required=True)
  p.add_argument(
    "--e3d",
    type=Path,
    default=Path("/home/thibault/npcc/results/e3d_unknown_structure"),
  )
  p.add_argument("--arm", default="npcc-tabpfn-quantiles-v3-probit")
  p.add_argument("--designs", nargs="+", default=["decay", "strong"])
  p.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  with np.load(a.e3d / "samples.npz") as f:
    samples = {k: f[k] for k in f.files if k.endswith("_logtrue")}
  rows: list[dict[str, object]] = []
  for path in sorted((a.e10 / "cells" / a.arm).glob("*_members.npy")):
    key = path.name.removesuffix("_members.npy")
    design, dim, seed = key.split("_")
    d = int(dim.removeprefix("d"))
    if design not in a.designs or d not in a.dims:
      continue
    logtrue = samples[f"{key}_logtrue"].astype(np.float64)
    mixture = np.load(a.e10 / "cells" / a.arm / f"{key}.npy").astype(np.float64)
    members = np.load(path).astype(np.float64)
    refs: dict[str, np.ndarray] = {"m1": members[0]}
    for m in COMPETITORS:
      refs[f"mixture:{m}"] = np.load(a.e10 / "cells" / m / f"{key}.npy")
    for arm in (*E3D_ARMS, f"{a.arm}-tll-ns"):
      file = a.e3d / "cells" / arm / f"{key}.npy"
      if file.is_file():
        refs[f"e3d:{arm}"] = np.load(file)
    base = {"design": design, "d": d, "seed": int(seed.removeprefix("s"))}
    rows += compare(base, mixture, members, refs, logtrue)
  cells, summary = summarize(rows)
  a.out.mkdir(parents=True, exist_ok=True)
  cells.to_csv(a.out / "compare_cells.csv", index=False)
  summary.to_csv(a.out / "compare.csv", index=False)
  with pd.option_context("display.width", 200):
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
  main()
