"""The full grid's design, pinned on a synthetic run.

Every number here was computed at ae3a246, the code the full grid ran at
(9f4068f) merged with its base branch, on a fixed synthetic run: the triple
with Td and w derived and a 5d design, out-of-fold training PITs, float32
quantiles, the (T, RH, p) subset and the projected log score. A change to
the code must leave its margins and copula scores as they were, so the run's
numbers stay what the merged code computes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.config import load_config
from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.runner import _marginal_set

#: On a fixed machine and thread count the outputs repeat bit for bit; this
#: tolerance is for another thread count, which moves the beta EMOS's
#: stopping point by about 1e-5 (its CRPS is flat at the minimum). A change
#: to the arithmetic moves these numbers far more.
_RTOL = 1e-4
_MODELS = ("emos-seasonal", "linear-gaussian-all")
_TRIPLE = [
  "obs:air_temperature",
  "obs:relative_humidity",
  "obs:surface_air_pressure",
]
_ALL = [
  "obs:air_temperature",
  "obs:dew_point_temperature",
  "obs:relative_humidity",
  "obs:surface_air_pressure",
  "obs:water_vapor_mixing_ratio",
]
#: The arms, run on the hermetic backend under TabPFN's name.
_ARMS = (
  "independence",
  "gaussian",
  "npcc-mixture-tabpfn-quantiles-time",
  "npcc-chain-tabpfn-quantiles-all",
)
_ARMS_5D = ("independence", "gaussian", "npcc-chain-tabpfn-quantiles-all")


def config_text(data_dir: Path, margins_extra: str = "") -> str:
  """The full grid's design on the hermetic backend, at one station."""
  return f"""
base_seed = 11

[data]
dir = "{data_dir}"
verify = false
stations = [0]
train = [2017-01-01, 2020-12-31]
test = [2021-01-01, 2021-12-31]

[margins]
models = {json.dumps(list(_MODELS))}
n_levels = 19
{margins_extra}
[responses]
modeled = {json.dumps(_TRIPLE)}
derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]
magnus = "meteoswiss"
"obs:air_temperature" = {{ transform = "identity" }}
"obs:dew_point_temperature" = {{ transform = "identity" }}
"obs:relative_humidity" = {{ transform = "logit", scale = 100.0, boundary_nudge = 0.05 }}
"obs:surface_air_pressure" = {{ transform = "identity" }}
"obs:water_vapor_mixing_ratio" = {{ transform = "log" }}

[copulas]
arms = ["independence", "gaussian", "independence@5d", "gaussian@5d"]
n_draws = 8
sample_batch_size = 50
mixture_k = 2
save_draws = true
subsets = {{ trhp = {json.dumps(_TRIPLE)} }}
projected_log_score = true

[copulas.designs.5d]
modeled = {json.dumps(_ALL)}
derived = []

[[estimators]]
label = "linear-gaussian-probit"
backend = "linear-gaussian"
transform = "probit"
"""


def compute(tmp_path: Path, margins_extra: str = "") -> dict[str, list[float]]:
  """Every pinned number of one synthetic run.

  The margins run through the command line; each copula arm runs on the
  marginal set the copula stage would read for its design, with the run's
  subsets and projection.
  """
  from npcc.experiments.application.copulas import run_copula_arm

  data = tmp_path / "data"
  write_synthetic(data, n_stations=1, step_days=7, seed=3, measured=True)
  path = tmp_path / "app.toml"
  path.write_text(config_text(data, margins_extra))
  run = tmp_path / "run"
  assert main(["margins", "--config", str(path), "--out", str(run)]) == 0
  config = load_config(path)
  spec = config.estimators_by_backend["linear-gaussian"]
  got: dict[str, list[float]] = {}
  for model in _MODELS:
    unit = run / "stations" / "000" / "margins" / model
    scores = pd.read_parquet(unit / "scores.parquet")
    means = scores.groupby("variable").mean(numeric_only=True)
    got[f"{model}:crps"] = means["crps"].tolist()
    got[f"{model}:nll"] = means["nll"].tolist()
    if "crps_closed_form" in means:
      got[f"{model}:crps_closed_form"] = means["crps_closed_form"].tolist()
    predictions = pd.read_parquet(unit / "predictions.parquet")
    train = predictions[predictions["split"] == "train"]
    got[f"{model}:pit_out_of_fold"] = (
      train.groupby("variable")["pit_out_of_fold"].mean().tolist()
    )
    for design, arms in ((None, _ARMS), ("5d", _ARMS_5D)):
      ms = _marginal_set(config, run, 0, model, design)
      got[f"{model}:{design}:q[7,[0,9,18]]"] = (
        ms.quantiles_test[7, [0, 9, 18]].ravel().tolist()
      )
      got[f"{model}:{design}:pit_train[3]"] = ms.pit_train[3].tolist()
      for arm in arms:
        for result in run_copula_arm(
          arm,
          ms,
          estimators={"tabpfn-quantiles": spec},
          n_draws=8,
          sample_batch_size=50,
          mixture_k=2,
          seed=7,
          device="cpu",
          sample_crps=design is not None,
          subsets={"trhp": tuple(_TRIPLE)},
          project_onto=None if design is None else tuple(_TRIPLE),
        ):
          row = [
            float(result.joint_nll.mean()),
            float(result.logcop_test[2]),
          ]
          if result.sampled:
            row += [
              float(result.energy.mean()),
              float(result.variogram.mean()),
              *(float(v.mean()) for _, v in sorted(result.sample_crps.items())),
            ]
          row += [float(v.mean()) for _, v in sorted(result.extra.items())]
          got[f"{model}:{design}:{result.arm}"] = row
  return got


#: The pinned numbers, by result and score, as :func:`compute` returns them.
PINNED: dict[str, list[float]] = json.loads(
  (Path(__file__).with_name("application_pinned.json")).read_text()
)


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_the_full_grids_design_reproduces_its_pinned_outputs(
  tmp_path: Path,
) -> None:
  """Its margins and copula scores are what they were when the run started."""
  pytest.importorskip("npcc.experiments.application.copulas")
  got = compute(tmp_path)
  assert set(got) == set(PINNED)
  for key, expected in PINNED.items():
    np.testing.assert_allclose(
      np.asarray(got[key], dtype=float),
      np.asarray(expected, dtype=float),
      rtol=_RTOL,
      err_msg=key,
    )
