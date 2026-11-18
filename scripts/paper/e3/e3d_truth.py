#!/usr/bin/env python3
"""E3d: the truths -- random R-vines on the nine Table 1 pair copulas.

Run in the NPCC environment, from the repository root:

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 10 nice -n 10 uv run --no-sync python scripts/paper/e3/e3d_truth.py

**E3d** asks what happens when the structure is unknown. E3c hands every arm
the true structure, an oracle; in practice a competitor must select one, and
a simplified fit on a wrong structure is inconsistent, whereas NPCC can fit a
non-simplified vine on any structure.

**Truths.** For ``d`` in ``{5, 10}`` and seeds ``101, 202, ..., 1010``:

- the structure is drawn uniformly over all R-vines on ``d`` variables with
  ``pyvinecopulib.RVineStructure.sample`` -- the sampler of npcc's vine study
  E2 -- seeded by ``derive_seed("E3d", "truth", d, seed)``;
- each edge takes one of the nine cases of VDC's Table 1 runner, drawn
  uniformly per edge, tree by tree, with ``np.random.default_rng([seed, d,
  4])``;
- two designs share that structure and those cases, as in E3c: ``decay``
  (the primary one), where an edge in tree ``t`` has Kendall's tau ``tau_case
  * 0.7 ** (t - 1)`` (``e3c_decay_vdc_side.decayed_bicop``), and ``strong``,
  every edge at its case's own parameter;
- the vine is simplified on its own structure, and the samples are
  ``Vinecop.simulate(2000, seeds=[seed, d, 11])`` and ``simulate(20000,
  seeds=[seed, d, 12])`` in pyvinecopulib 1.0 (the version with E2's sampler;
  E3a-c drew with 0.7.5), clipped to ``[1e-10, 1 - 1e-10]``, with the true
  log-density of every test point.

**Random structure.** NPCC's random-structure arms use a structure drawn the
same way, independently of the truth, seeded by ``derive_seed("E3d",
"random", d, seed)`` -- as E2's ``random`` arm draws its own.

**Outputs** (``results/e3d_unknown_structure/``): ``samples.npz`` (per design
and cell: train, test, true log-density, the true and the random structure
matrices, the edge cases and each edge's tau) and ``truths.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402
import e3d_structure  # noqa: E402
from e3a_vdc_side import case_label, runner_constants  # noqa: E402
from e3c_decay_vdc_side import decayed_bicop  # noqa: E402
from npcc.experiments._study import derive_seed  # noqa: E402

RESULTS = common.NPCC_REPO / "results" / "e3d_unknown_structure"
DIMS = (5, 10)
SEEDS = tuple(101 * k for k in range(1, 11))
DESIGNS = ("decay", "strong")
N_TRAIN = 2000
N_TEST = 20000


def case_bicop(case: tuple):  # noqa: ANN201 - a pyvinecopulib Bicop
  """The case's pair copula, at its own parameter."""
  family, params, rotation = case
  if family == "indep":
    return pv.Bicop()
  return pv.Bicop(family=getattr(pv.BicopFamily, family), rotation=rotation,
                  parameters=np.asarray(params))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--out", type=Path, default=RESULTS)
  a = p.parse_args()

  e3d_structure.self_check(pv)
  cases, _ = runner_constants(common.TABLE1_RUNNER.read_text())
  a.out.mkdir(parents=True, exist_ok=True)
  prov = common.provenance(__file__, "cpu")

  arrays: dict[str, np.ndarray] = {}
  index = {"design": [], "d": [], "seed": []}
  truths = []
  for d in DIMS:
    for seed in SEEDS:
      truth = pv.RVineStructure.sample(d, seeds=[derive_seed("E3d", "truth", d, seed)])
      random = pv.RVineStructure.sample(d, seeds=[derive_seed("E3d", "random", d, seed)])
      rng = np.random.default_rng([seed, d, 4])
      edge_cases = [[int(rng.integers(len(cases))) for _ in range(d - 1 - t)] for t in range(d - 1)]
      truth_edges = e3d_structure.edges_from_matrix(np.asarray(truth.matrix))
      rand_rec = e3d_structure.recovery(e3d_structure.edges_from_matrix(np.asarray(random.matrix)),
                                        truth_edges)
      for design in DESIGNS:
        if design == "decay":
          made = [[decayed_bicop(pv, cases[j], t) for j in row] for t, row in enumerate(edge_cases)]
          pcs = [[bc for bc, _ in row] for row in made]
          taus = [tau for row in made for _, tau in row]
        else:
          pcs = [[case_bicop(cases[j]) for j in row] for row in edge_cases]
          taus = [float(bc.parameters_to_tau(bc.parameters)) for row in pcs for bc in row]
        vine = pv.Vinecop.from_structure(structure=truth, pair_copulas=pcs)
        tr = np.clip(vine.simulate(N_TRAIN, seeds=[seed, d, 11]), 1e-10, 1 - 1e-10)
        te = np.clip(vine.simulate(N_TEST, seeds=[seed, d, 12]), 1e-10, 1 - 1e-10)
        logtrue = np.log(np.maximum(np.asarray(vine.pdf(te)), common.LOG_FLOOR))
        if not np.all(np.isfinite(logtrue)):
          raise RuntimeError(f"{design} d={d} seed={seed}: the true log-density must be finite")
        key = f"{design}_d{d}_s{seed}"
        arrays.update({f"{key}_train": tr, f"{key}_test": te, f"{key}_logtrue": logtrue,
                       f"{key}_true_matrix": np.asarray(truth.matrix),
                       f"{key}_random_matrix": np.asarray(random.matrix),
                       f"{key}_edge_cases": np.asarray([c for row in edge_cases for c in row]),
                       f"{key}_edge_taus": np.asarray(taus)})
        index["design"].append(design); index["d"].append(d); index["seed"].append(seed)
        truths.append({
          "design": design, "d": d, "seed": seed,
          "true_matrix": np.asarray(truth.matrix).tolist(),
          "random_matrix": np.asarray(random.matrix).tolist(),
          "random_vs_truth": rand_rec,
          "edge_cases": [[case_label(*cases[j]) for j in row] for row in edge_cases],
          "edge_taus": [round(t, 6) for t in taus],
          "true_loglik": float(np.mean(logtrue)),
        })
        print(f"{key}: true ll {np.mean(logtrue):.4f}; random structure recovers "
              f"{rand_rec['overall']:.2f} of the true edges", flush=True)

  samples = a.out / "samples.npz"
  common.save_npz(samples, {"design": np.asarray(index["design"]), "d": np.asarray(index["d"]),
                            "seed": np.asarray(index["seed"]), **arrays})
  common.write_json({
    "experiment": "E3d", "description": "unconditional vines with an unknown structure: the truths",
    "provenance": prov,
    "design_spec": {"dims": list(DIMS), "seeds": list(SEEDS), "designs": list(DESIGNS),
                    "n_train": N_TRAIN, "n_test": N_TEST,
                    "structure": "pv.RVineStructure.sample(d, seeds=[derive_seed('E3d', 'truth', d, seed)])",
                    "random_structure": "pv.RVineStructure.sample(d, seeds=[derive_seed('E3d', 'random', d, seed)])",
                    "edges": "case index rng.integers(9) per edge, tree by tree, rng = default_rng([seed, d, 4])",
                    "decay": "tau_case * 0.7 ** (t - 1) (e3c_decay_vdc_side.decayed_bicop)",
                    "simulate": "pyvinecopulib 1.0, seeds [seed, d, 11] (train), [seed, d, 12] (test)",
                    "sample_clip": [1e-10, 1 - 1e-10], "cases": cases},
    "truths": truths,
    "samples_npz": samples.name, "samples_sha256": common.sha256_file(samples),
  }, a.out / "truths.json")


if __name__ == "__main__":
  main()
