"""Refit E1's models at n = 1000 and evaluate exact conditional density slices.

E1 stores its surfaces on a 20 x 20 midpoint grid, which holds no slice at
u1 = 0.25, 0.5 or 0.75. This script refits the chosen estimators on E1's exact
data -- the same cell seed, ``scenarios.sample`` draw and per-estimator torch
seed the runner uses, so the models equal E1's up to GPU nondeterminism -- and
evaluates c(u1, u2 | x) along fine u2 grids at the x where tau(x) hits each
target level (the first such x in the sine regime).

Two steps, so the GPU part can be split across processes:

- ``fit``: writes ``slices_<part>.csv`` (long format: one row per family x
  regime x rep x estimator x target tau x u1 x u2, with the truth) into
  ``--out``. ``--part I/K`` takes every K-th (family, regime, rep) cell.
- ``plot``: reads every ``slices_*.csv`` and draws
  ``density_recovery_grid_icl.pdf`` and ``density_recovery_grid_gb.pdf`` (and
  PNGs): rows are target tau x u1, columns family x regime; each backend is its
  mean curve over reps with a light band for the pointwise 10-90% range, and
  the truth in black.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e1/density_slices.py fit \
      --tabpfn v3 --transform probit --out results/study_2026-11-05/analysis/slices
    uv run --no-sync python scripts/paper/e1/density_slices.py plot \
      --out results/study_2026-11-05/analysis/slices
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import BACKENDS, LINESTYLE, _style, annotate, save

from npcc.core import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments import scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import (
  Cell,
  EstimatorSpec,
  GridConfig,
  load_grid,
)
from npcc.experiments.runner import _cell_seed, _surface_x_rows

PANELS = [(f, s) for f in ("clayton", "gaussian") for s in ("linear", "sin")]


def _part(text: str) -> tuple[int, int]:
  index, _, count = text.partition("/")
  return int(index), int(count)


def chosen_estimators(
  grid: GridConfig, tabpfn: str, transform: str
) -> list[EstimatorSpec]:
  """The six backends at one TabPFN version and transform, in BACKENDS order."""
  order = list(BACKENDS)
  picked: list[tuple[int, EstimatorSpec]] = []
  for est in grid.estimators:
    meta = annotate(pd.DataFrame({"label": [est.label]})).iloc[0]
    if meta["transform_"] != transform:
      continue
    if meta["key"] == "tabpfn" and meta["version"] != tabpfn:
      continue
    picked.append((order.index(meta["key"]), est))
  return [est for _, est in sorted(picked, key=lambda p: p[0])]


def fit(a: argparse.Namespace) -> None:
  grid = load_grid(a.config)
  estimators = chosen_estimators(grid, a.tabpfn, a.transform)
  cells = [
    Cell(family=f, tau_scenario=s, n=a.n, rep=r)
    for f, s in PANELS
    for r in range(grid.n_rep)
  ]
  index, count = a.part
  cells = [c for i, c in enumerate(cells) if i % count == index]
  u2 = torch.linspace(0.0, 1.0, a.points + 2, dtype=torch.float64)[1:-1]

  commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
  ).stdout.strip()
  import tabpfn

  rows = []
  for cell in cells:
    seed = _cell_seed(a.base_seed, cell)
    u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
    targets = {}
    for row in _surface_x_rows(cell.tau_scenario, list(a.tau)):
      targets.setdefault(row["target_tau"], row)
    fam = scenarios.FAMILIES[cell.family]
    for est in estimators:
      torch.manual_seed(derive_seed(seed, est.estimator_id))
      model = RosenblattBicop(
        est.controls(
          FitControlsRosenblattBicop,
          device=a.device,
          projection_grid_size=grid.projection_grid_size,
          batch_size=grid.batch_size,
        )
      )
      model.fit(torch.column_stack([u, v]), x=x)
      for target, row in targets.items():
        cop = scenarios.bicop(fam, row["tau_true"])
        for u1 in a.u1:
          uv = torch.column_stack([torch.full_like(u2, u1), u2])
          xs = torch.full((u2.shape[0], 1), row["x"], dtype=torch.float64)
          with torch.inference_mode():
            pred = model.pdf(uv, x=xs).cpu().numpy()
          truth = cop.pdf(uv.numpy())
          rows.append(
            pd.DataFrame(
              {
                "family": cell.family,
                "tau_scenario": cell.tau_scenario,
                "n": cell.n,
                "rep": cell.rep,
                "cell_seed": seed,
                "label": est.label,
                "estimator_id": est.estimator_id,
                "target_tau": target,
                "x": row["x"],
                "tau_true": row["tau_true"],
                "u1": u1,
                "u2": u2.numpy(),
                "pdf_hat": pred,
                "pdf_true": truth,
                "npcc_commit": commit,
                "tabpfn_version": tabpfn.__version__,
                "device": str(model._device),
              }
            )
          )
    print(f"done {cell.family}/{cell.tau_scenario} rep {cell.rep}", flush=True)
  a.out.mkdir(parents=True, exist_ok=True)
  pd.concat(rows, ignore_index=True).to_csv(
    a.out / f"slices_{index}of{count}.csv", index=False
  )


def plot(a: argparse.Namespace) -> None:
  import matplotlib.pyplot as plt

  parts = sorted(glob.glob(str(a.out / "slices_*.csv")))
  if not parts:
    raise SystemExit(f"no slices_*.csv in {a.out}")
  df = annotate(pd.concat((pd.read_csv(p) for p in parts), ignore_index=True))
  _style()
  levels = sorted(df["target_tau"].unique())
  u1s = sorted(df["u1"].unique())
  for group, name in (("ICL", "icl"), ("GB", "gb")):
    sub = df[df["group"] == group]
    if sub.empty:
      print(f"no {group} estimators in the slices; skipped {name}")
      continue
    labels = [
      lab
      for lab in dict.fromkeys(
        sub.sort_values("key", key=lambda k: k.map(list(BACKENDS).index))[
          "label"
        ]
      )
    ]
    nrows = len(levels) * len(u1s)
    fig, axes = plt.subplots(
      nrows,
      len(PANELS),
      figsize=(7.0, 1.35 * nrows + 0.6),
      sharex=True,
      sharey=False,
      constrained_layout=True,
    )
    for r, (target, u1) in enumerate((t, u) for t in levels for u in u1s):
      for c, (family, scenario) in enumerate(PANELS):
        ax = axes[r, c]
        panel = sub[
          (sub["family"] == family)
          & (sub["tau_scenario"] == scenario)
          & (sub["target_tau"] == target)
          & (sub["u1"] == u1)
        ]
        if r == 0:
          ax.set_title(f"{family.capitalize()}, {scenario}", fontsize=9)
        if c == 0:
          ax.set_ylabel(f"$\\tau={target:g}$\n$u_1={u1:g}$", fontsize=8)
        if panel.empty:
          continue
        truth = panel.groupby("u2")["pdf_true"].first()
        # Each panel is read against its own truth, so its scale is the
        # truth's: curves above 1.5 times its maximum are clipped.
        ax.set_ylim(0.0, 1.5 * float(truth.max()))
        ax.plot(
          truth.index,
          truth.to_numpy(),
          color="black",
          linewidth=1.8,
          label="Truth",
          zorder=5,
        )
        for label in labels:
          rows = panel[panel["label"] == label]
          if rows.empty:
            continue
          spec = BACKENDS[rows["key"].iloc[0]]
          g = rows.groupby("u2")["pdf_hat"]
          mean, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
          ax.fill_between(
            mean.index,
            lo.to_numpy(),
            hi.to_numpy(),
            color=spec["color"],
            alpha=0.18,
            linewidth=0,
          )
          ax.plot(
            mean.index,
            mean.to_numpy(),
            color=spec["color"],
            linestyle=LINESTYLE[spec["group"]],
            linewidth=1.3,
            label=rows["display"].iloc[0],
          )
    for ax in axes[-1]:
      ax.set_xlabel("$u_2$")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=len(names))
    save(fig, a.out / f"density_recovery_grid_{name}.pdf")
  print(f"wrote density_recovery_grid_{{icl,gb}}.pdf from {len(parts)} part(s)")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  sub = p.add_subparsers(dest="step", required=True)
  f = sub.add_parser("fit")
  f.add_argument("--config", type=Path, default=Path("configs/study.toml"))
  f.add_argument("--tabpfn", required=True, help="Main-text TabPFN version.")
  f.add_argument("--transform", default="probit", help="Main-text transform.")
  f.add_argument("--n", type=int, default=1000)
  f.add_argument("--u1", type=float, nargs="+", default=[0.25, 0.5, 0.75])
  f.add_argument("--tau", type=float, nargs="+", default=[0.4, 0.7])
  f.add_argument("--points", type=int, default=200)
  f.add_argument("--base-seed", type=int, default=317)
  f.add_argument("--device", default=None)
  f.add_argument("--part", type=_part, default=(0, 1), metavar="I/K")
  f.add_argument("--out", type=Path, required=True)
  g = sub.add_parser("plot")
  g.add_argument("--out", type=Path, required=True)
  a = p.parse_args()
  if a.step == "fit":
    fit(a)
  else:
    plot(a)


if __name__ == "__main__":
  main()
