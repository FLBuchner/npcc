"""The default five-response design, pinned before the triple was added.

Every number here was computed at f1716a9, before the modeled-subset option
existed, on a fixed synthetic run (on four CPU cores, where the new code
reproduces them bit for bit): a code path added for another design must leave
the default's predictions and copula scores as they were, and its
configuration signature, so a run folder started earlier still resumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.config import load_config
from npcc.experiments.application.store import read_marginal_set
from tests.experiments._application import write_config, write_data

_ROOT = Path(__file__).resolve().parents[2]
#: On a fixed machine and thread count the outputs repeat bit for bit; this
#: tolerance is for another thread count. The beta EMOS's stopping point
#: moves by about 1e-5 with it -- its CRPS is flat at the minimum and its
#: gradient is a finite difference, which BLAS's summation order perturbs --
#: and everything downstream of the humidity margin moves with it. A change
#: to the default path's arithmetic moves these numbers far more.
_RTOL = 1e-4
_ARMS = (
  "independence",
  "gaussian",
  "npcc-random-tabpfn-quantiles-time",
  "npcc-mixture-tabpfn-quantiles-time",
)

PINNED: dict[str, list[object]] = {
  "emos-seasonal:pit_test[3]": [
    0.7805372094541166,
    0.3402696541207837,
    0.8139152872308713,
    0.2712797746898733,
    0.32504118822814837,
  ],
  "emos-seasonal:logpdf_test[5]": [
    -1.9492158433747577,
    -2.4907909825714456,
    -3.7046567863190987,
    -1.7815204597599097,
    -1.058220084761448,
  ],
  "emos-seasonal:q[7,[0,9,18]]": [
    [
      -4.906005935696223,
      -6.7332925098398295,
      56.62521621114194,
      954.4442246275299,
      1.5977794081387955,
    ],
    [
      -1.0205736478816378,
      -2.1033627535338897,
      86.91010923781425,
      957.8674550202384,
      3.333572644182454,
    ],
    [
      2.864858639932949,
      2.5265670027720533,
      98.81116908878,
      961.2906854129468,
      5.070609141993355,
    ],
  ],
  "emos-seasonal:crps_mean": [
    1.125822044548507,
    1.417936697802093,
    9.780685000982116,
    1.0649517568793203,
    0.5772098001701087,
  ],
  "emos-seasonal:independence": [
    14.544741079846036,
    0.6005123027616188,
    0.4001122716190672,
    0.0,
  ],
  "emos-seasonal:gaussian": [
    14.593676042645189,
    0.6107787173414392,
    0.3953130218101735,
    0.2384127369261238,
  ],
  "emos-seasonal:npcc-random-tabpfn-quantiles-time": [
    13.65503386014631,
    0.6059501008333421,
    0.4357978180338655,
    0.021324409338569106,
  ],
  "emos-seasonal:npcc-mixture-tabpfn-quantiles-time": [
    11.935220324108876,
    0.629222108025011,
    0.42544873377516657,
    -0.08454803005749079,
  ],
  "emos-seasonal:npcc-mixture-tabpfn-quantiles-time-m1": [
    12.016052923688147,
    0.6653881920692097,
    0.4433196415133914,
    -0.31272767829849896,
  ],
  "emos-seasonal:npcc-mixture-tabpfn-quantiles-time-m2": [
    13.628325264442962,
    0.6124747472604442,
    0.4284182120168079,
    0.10111688949971473,
  ],
  "linear-gaussian-all:pit_test[3]": [
    0.4480841371585081,
    0.7474562083669247,
    0.4215725548243569,
    0.2925210706202519,
    0.043385067467942656,
  ],
  "linear-gaussian-all:logpdf_test[5]": [
    -1.6818932992065507,
    -2.0967733729843103,
    -5.095696827266428,
    -1.7709741737018723,
    -0.3359442420224015,
  ],
  "linear-gaussian-all:q[7,[0,9,18]]": [
    [
      -4.137204564577649,
      -6.299192935176554,
      2.730739292045929,
      954.6761514185798,
      2.678619086408286,
    ],
    [
      -0.6528441772052553,
      -2.2013329446613907,
      91.13387398421897,
      958.0407160990793,
      3.4082389640009643,
    ],
    [
      2.8315162101670697,
      1.8965270458536527,
      99.97343576109323,
      961.4052807796018,
      4.336597500807957,
    ],
  ],
  "linear-gaussian-all:crps_mean": [
    1.0667901227517809,
    1.1338640174112458,
    13.908713452442154,
    1.0761296030532983,
    0.4349814752079929,
  ],
  "linear-gaussian-all:independence": [
    11.651213094031274,
    0.7422154891159666,
    0.522200133104539,
    0.0,
  ],
  "linear-gaussian-all:gaussian": [
    11.465510061772815,
    0.6916453903115403,
    0.47092523677575093,
    0.6885509112613231,
  ],
  "linear-gaussian-all:npcc-random-tabpfn-quantiles-time": [
    11.44066163450889,
    0.6088751008838227,
    0.4441191513969712,
    0.8281940607107225,
  ],
  "linear-gaussian-all:npcc-mixture-tabpfn-quantiles-time": [
    11.43923795485755,
    0.7209919856941155,
    0.5070450369422346,
    0.8498215951369522,
  ],
  "linear-gaussian-all:npcc-mixture-tabpfn-quantiles-time-m1": [
    11.436024360793947,
    0.7536051315684831,
    0.5607806569778893,
    0.852667678409061,
  ],
  "linear-gaussian-all:npcc-mixture-tabpfn-quantiles-time-m2": [
    11.442611929903494,
    0.7540108158201808,
    0.5170500395072539,
    0.846967388549721,
  ],
}


def test_the_shipped_default_configs_keep_their_signature() -> None:
  """A changed signature would refuse to resume a folder the default started."""
  for name in ("application.toml", "application_smoke.toml"):
    config = load_config(_ROOT / "configs" / name)
    assert config.signature() == (
      "7e8253a66d39b035ee1902bd01f3a8c66a7bf9821343bf6ea797eb86b5833899"
    )
    assert config.modeled == config.variables


def test_the_default_design_reproduces_its_pinned_outputs(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """The default path's margins and copula scores are what they were.

  The copula stage is the real one; its npcc arms run the hermetic backend
  under TabPFN's name.
  """
  del register_linear_gaussian_backend
  copulas = pytest.importorskip("npcc.experiments.application.copulas")
  config_path = write_config(
    tmp_path / "app.toml",
    write_data(tmp_path / "data"),
    stations="[0]",
    models=["emos-seasonal", "linear-gaussian-all"],
  )
  run = tmp_path / "run"
  assert main(["margins", "--config", str(config_path), "--out", str(run)]) == 0
  config = load_config(config_path)
  spec = config.estimators_by_backend["linear-gaussian"]
  got: dict[str, list[object]] = {}
  for model in ("emos-seasonal", "linear-gaussian-all"):
    ms = read_marginal_set(run, 0, model)
    assert ms.derived == ()
    got[f"{model}:pit_test[3]"] = ms.pit_test[3].tolist()
    got[f"{model}:logpdf_test[5]"] = ms.logpdf_test[5].tolist()
    got[f"{model}:q[7,[0,9,18]]"] = ms.quantiles_test[7, [0, 9, 18]].tolist()
    scores = pd.read_parquet(
      run / "stations" / "000" / "margins" / model / "scores.parquet"
    )
    got[f"{model}:crps_mean"] = (
      scores.groupby("variable")["crps"].mean().tolist()
    )
    for arm in _ARMS:
      for result in copulas.run_copula_arm(
        arm,
        ms,
        estimators={"tabpfn-quantiles": spec},
        n_draws=8,
        sample_batch_size=50,
        mixture_k=2,
        seed=7,
        device="cpu",
      ):
        assert set(result.sample_crps) == set()
        got[f"{model}:{result.arm}"] = [
          float(result.joint_nll.mean()),
          float(result.energy.mean()),
          float(result.variogram.mean()),
          float(result.logcop_test[2]),
        ]
  assert set(got) == set(PINNED)
  for key, expected in PINNED.items():
    np.testing.assert_allclose(
      np.asarray(got[key], dtype=float),
      np.asarray(expected, dtype=float),
      rtol=_RTOL,
      err_msg=key,
    )
