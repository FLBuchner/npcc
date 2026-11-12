"""E12: the total mass of E3a's pair copulas and E3c's vines, per cell.

The same estimate as ``mass_e3.py`` (``mass_common.stats``), on the two E3
studies whose test points are drawn from a known truth:

- **E3a**, single pair copulas (d = 2): every NPCC estimator and VDC's
  adapted pair, at ``n_train`` 2,000 and 10,000. Both sample sizes share
  the test points and their true log-density, ``e3a/samples.npz``'s
  ``logtrue``.
- **E3c**, vines on the true structure, simplified, on both designs
  (``e3c_decay`` and ``e3c``, the ``strong`` one): the NPCC arms, TLL,
  parametric and VDC's pairs in the shared vine (``npcc_cells/``), and VDC's
  own vine (``vdc_own_cells/``, arm ``vdc_own``).

Test points where the fitted log-density is not finite, or at most
``log(1e-300) + 1`` (the floor E3 writes for a zero density), are left out
and counted in ``n_excl``.

**Inputs.** ``--e3-vdc``, E3's folder (``results/e3_vdc``).

**Outputs.** ``mass_e3ac_cells.csv`` in ``--out``: one row per study, arm,
design, d, ``n_train`` and seed, with ``mass_common.stats``, ``n_excl`` and
``json_kl``, the cell row's KL (E3a) or excess NLL (E3c). It also prints the
mean mass per E3a arm and ``n_train`` (overall and per case), and per E3c
design, d and arm, where the estimate's columns read ``not estimable`` at
``mass_common.NOT_ESTIMABLE``.

**Usage** (from the repository root, on the CPU, in seconds)::

    uv run --no-sync python scripts/paper/e12/mass_e3ac.py --out results/e12_<TAG>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mass_common import RESULTS, printable, stats

#: E3's floor for a zero density, as a log.
FLOOR = np.log(1e-300)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--e3-vdc", type=Path, default=RESULTS / "e3_vdc")
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()
  res = a.e3_vdc
  a.out.mkdir(parents=True, exist_ok=True)

  rows = []
  with np.load(res / "e3a" / "samples.npz") as f:
    lt_all = f["logtrue"].astype(np.float64)
    case = f["case"]
    seed = f["seed"]
  for est_dir in sorted((res / "e3a" / "npcc_cells").iterdir()):
    for ndir in sorted(est_dir.iterdir()):
      for npy in sorted(ndir.glob("*.npy")):
        k = int(npy.stem.split("_")[0][1:])
        lp = np.load(npy).astype(np.float64)
        fin = np.isfinite(lp) & (lp > FLOOR + 1)
        lr = lp[fin] - lt_all[k][fin]
        j = json.loads(npy.with_suffix(".json").read_text())
        rows.append(
          {
            "study": "E3a",
            "arm": est_dir.name,
            "design": str(case[k]),
            "d": 2,
            "n_train": int(ndir.name[1:]),
            "seed": int(seed[k]),
            "n_excl": int((~fin).sum()),
            "json_kl": j.get("kl"),
            **stats(lr),
          }
        )
  for design, folder in (("decay", "e3c_decay"), ("strong", "e3c")):
    with np.load(res / folder / "samples.npz") as f:
      lt = {
        k.removesuffix("_logtrue"): f[k].astype(np.float64)
        for k in f.files
        if k.endswith("_logtrue")
      }
    for sub in ("npcc_cells", "vdc_own_cells"):
      base = res / folder / sub
      if not base.is_dir():
        continue
      if sub == "npcc_cells":
        arm_dirs = sorted(q for q in base.iterdir() if q.is_dir())
      else:
        arm_dirs = [base]
      for arm_dir in arm_dirs:
        arm = arm_dir.name if sub == "npcc_cells" else "vdc_own"
        for npy in sorted(arm_dir.glob("*.npy")):
          key = npy.stem
          if key not in lt:
            continue
          lp = np.load(npy).astype(np.float64)
          fin = np.isfinite(lp) & (lp > FLOOR + 1)
          lr = lp[fin] - lt[key][fin]
          js = npy.with_suffix(".json")
          jx = (
            json.loads(js.read_text()).get("excess_nll")
            if js.exists()
            else None
          )
          rows.append(
            {
              "study": "E3c",
              "arm": arm,
              "design": design,
              "d": int(key.split("_")[0][1:]),
              "n_train": 2000,
              "seed": int(key.split("_")[1][1:]),
              "n_excl": int((~fin).sum()),
              "json_kl": jx,
              **stats(lr),
            }
          )
  df = pd.DataFrame(rows)
  df.to_csv(a.out / "mass_e3ac_cells.csv", index=False)

  pd.set_option("display.width", 250)
  pd.set_option("display.max_rows", 300)
  e3a = df[df.study == "E3a"]
  by_arm = e3a.groupby(["arm", "n_train"]).agg(
    n=("mass", "size"),
    mass=("mass", "mean"),
    mn=("mass", "min"),
    mx=("mass", "max"),
    se_med=("mass_se", "median"),
    mom=("mass_mom20", "mean"),
    neglogm=("neglogmass", "mean"),
    kl=("excess_nll", "mean"),
    json_kl=("json_kl", "mean"),
  )
  print(by_arm.to_string(float_format=lambda x: f"{x:.4f}"))
  by_case = e3a.groupby(["arm", "n_train", "design"]).agg(
    mass=("mass", "mean"), mn=("mass", "min"), mx=("mass", "max")
  )
  print(by_case.to_string(float_format=lambda x: f"{x:.4f}"))
  e3c = (
    df[df.study == "E3c"]
    .groupby(["design", "d", "arm"])
    .agg(
      n=("mass", "size"),
      mass=("mass", "mean"),
      mn=("mass", "min"),
      mx=("mass", "max"),
      se_med=("mass_se", "median"),
      mom=("mass_mom20", "mean"),
      trim=("mass_trim_top0.1pct", "mean"),
      ess_min=("ess_share", "min"),
      neglogm=("neglogmass", "mean"),
      excess=("excess_nll", "mean"),
      json_excess=("json_kl", "mean"),
      excl=("n_excl", "sum"),
    )
  )
  is_columns = ["mass", "mn", "mx", "se_med", "mom", "trim", "neglogm"]
  print(printable(e3c, is_columns).to_string())


if __name__ == "__main__":
  main()
