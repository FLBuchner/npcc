"""Check that E2's truths are valid at a dimension, on the CPU.

For each repetition's truth, as ``npcc-vinestudy`` draws it, this reports:

- Kendall's tau per tree over a fine x grid, against E2's bands;
- the training samples at each sample size: whether every ``y`` is finite, and
  how many PITs reach the ``1e-9`` clamp, which is where the inverse
  Rosenblatt recursion saturates;
- the test points: whether every true joint and copula density is finite and
  positive, the range of the log copula density, and the mean true
  log-density at each evaluation x.

It prints one JSON line per repetition, then a summary over them. Usage (from
the npcc repository root)::

    uv run --no-sync python scripts/paper/e2/check_truths.py --d 10 --reps 20
"""

from __future__ import annotations

import argparse
import json

import torch

from npcc.experiments import scenarios, vine_runner, vine_scenarios
from npcc.experiments._study import derive_seed

#: E2's settings: base seed, sample sizes, evaluation x and draws per x.
BASE_SEED = 317
N = (500, 1000)
EVAL_X_N = 5
EVAL_M = 1000
#: The clamp ``vine_scenarios`` applies to every PIT.
EPS = 1e-9


def at_clamp(u: torch.Tensor) -> int:
  """How many PITs sit at the clamp."""
  return int(((u <= EPS) | (u >= 1.0 - EPS)).sum())


def check(d: int, rep: int) -> dict[str, object]:
  """One repetition's truth, its training samples and its test points."""
  truth = vine_runner.truth_for(BASE_SEED, d, rep)
  x_grid = torch.linspace(
    scenarios.X_MIN, scenarios.X_MAX, 101, dtype=torch.float64
  )
  bands: dict[int, list[float]] = {}
  for row in truth.edges:
    for edge in row:
      tau = vine_scenarios.edge_tau(edge.regime, edge.tree, x_grid)
      lo, hi = bands.get(edge.tree, [1.0, -1.0])
      bands[edge.tree] = [min(lo, float(tau.min())), max(hi, float(tau.max()))]
  record: dict[str, object] = {"rep": rep, "tau_by_tree": bands}
  for n in N:
    y, u, _ = vine_scenarios.sample(
      truth, n, derive_seed(BASE_SEED, "train", d, n, rep)
    )
    record[f"train_n{n}"] = {
      "y_finite": bool(torch.isfinite(y).all()),
      "u_at_clamp": at_clamp(u),
      "u_min": float(u.min()),
      "one_minus_u_min": float(1.0 - u.max()),
    }
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(EVAL_X_N),
    EVAL_M,
    derive_seed(BASE_SEED, "eval", d, rep),
  )
  copula = torch.cat(points.copula_pdf_true)
  joint = torch.cat(points.pdf_true)
  record["eval"] = {
    "pdf_finite_positive": bool((torch.isfinite(joint) & (joint > 0)).all()),
    "copula_pdf_finite_positive": bool(
      (torch.isfinite(copula) & (copula > 0)).all()
    ),
    "u_at_clamp": at_clamp(torch.cat(points.u)),
    "log_copula_pdf_range": [
      float(copula.log().min()),
      float(copula.log().max()),
    ],
    "loglik_true_by_x": [float(p.log().mean()) for p in points.pdf_true],
  }
  return record


def summary(records: list[dict]) -> dict[str, object]:
  """Each check over every repetition."""
  bands: dict[int, list[float]] = {}
  for r in records:
    for tree, (lo, hi) in r["tau_by_tree"].items():
      a, b = bands.get(tree, [1.0, -1.0])
      bands[tree] = [min(a, lo), max(b, hi)]
  evals = [r["eval"] for r in records]
  out: dict[str, object] = {
    "reps": len(records),
    "tau_by_tree": bands,
    "test_points": {
      "pdf_finite_positive": all(e["pdf_finite_positive"] for e in evals),
      "copula_pdf_finite_positive": all(
        e["copula_pdf_finite_positive"] for e in evals
      ),
      "u_at_clamp": sum(e["u_at_clamp"] for e in evals),
      "log_copula_pdf_range": [
        min(e["log_copula_pdf_range"][0] for e in evals),
        max(e["log_copula_pdf_range"][1] for e in evals),
      ],
      "loglik_true_range": [
        min(v for e in evals for v in e["loglik_true_by_x"]),
        max(v for e in evals for v in e["loglik_true_by_x"]),
      ],
    },
  }
  for n in N:
    train = [r[f"train_n{n}"] for r in records]
    out[f"train_n{n}"] = {
      "y_finite": all(t["y_finite"] for t in train),
      "u_at_clamp": sum(t["u_at_clamp"] for t in train),
      "u_min": min(t["u_min"] for t in train),
      "one_minus_u_min": min(t["one_minus_u_min"] for t in train),
    }
  return out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--d", type=int, required=True)
  p.add_argument("--reps", type=int, default=20)
  a = p.parse_args()
  records = []
  for rep in range(a.reps):
    records.append(check(a.d, rep))
    print(json.dumps(records[-1]), flush=True)
  print(json.dumps({"summary": summary(records)}, indent=2))


if __name__ == "__main__":
  main()
