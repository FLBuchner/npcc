from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from npcc.experiments.plots import metric_boxplot_by_n


def test_metric_boxplot_by_n_groups_repetitions_by_model_and_n() -> None:
  rows = []
  for n in (100, 200):
    for label in ("lightgbm-logit", "tabpfn-quantiles-logit"):
      for rep in (0, 1, 2):
        for x in (0.25, 0.75):
          rows.append(
            {
              "family": "clayton",
              "tau_scenario": "linear",
              "quantity": "pdf",
              "n": n,
              "label": label,
              "normalize": "none",
              "rep": rep,
              "x": x,
              "ISE": n / 100 + rep + x,
            }
          )
  metrics_df = pd.DataFrame(rows)

  ax = metric_boxplot_by_n(
    metrics_df,
    family="clayton",
    tau_scenario="linear",
    quantity="pdf",
    metric="ISE",
  )

  assert [tick.get_text() for tick in ax.get_xticklabels()] == ["100", "200"]
  assert ax.get_xlabel() == "Sample size"
  assert ax.get_ylabel() == "ISE"
  assert len(ax.patches) == 4
  legend = ax.get_legend()
  assert legend is not None
  legend_anchor = legend.get_bbox_to_anchor().transformed(
    ax.transAxes.inverted()
  )
  assert legend_anchor.x0 == pytest.approx(1.02)
  assert [text.get_text() for text in legend.get_texts()] == [
    "label=lightgbm-logit, normalize=none",
    "label=tabpfn-quantiles-logit, normalize=none",
  ]


def test_metric_boxplot_by_n_ignores_missing_metric_values() -> None:
  metrics_df = pd.DataFrame(
    {
      "family": ["clayton", "clayton"],
      "tau_scenario": ["constant", "constant"],
      "quantity": ["pdf", "pdf"],
      "n": [100, 100],
      "label": ["lightgbm-logit", "lightgbm-logit"],
      "normalize": ["none", "none"],
      "rep": [0, 1],
      "IAE": [0.2, float("nan")],
    }
  )

  ax = metric_boxplot_by_n(
    metrics_df,
    family="clayton",
    tau_scenario="constant",
    quantity="pdf",
    metric="IAE",
  )

  median = ax.lines[4].get_ydata()
  assert median == pytest.approx([0.2, 0.2])


def test_metric_boxplot_by_n_filters_each_model_attribute() -> None:
  metrics_df = pd.DataFrame(
    [
      {
        "family": "clayton",
        "tau_scenario": "constant",
        "quantity": "pdf",
        "n": 100,
        "label": label,
        "normalize": normalize,
        "rep": 0,
        "ISE": 0.1,
      }
      for label in ("lightgbm-logit", "tabpfn-quantiles-logit")
      for normalize in ("none", "5")
    ]
  )

  ax = metric_boxplot_by_n(
    metrics_df,
    family="clayton",
    tau_scenario="constant",
    quantity="pdf",
    metric="ISE",
    label=["lightgbm-logit", "tabpfn-quantiles-logit"],
    normalize="5",
  )

  assert len(ax.patches) == 2
  legend = ax.get_legend()
  assert legend is not None
  assert [text.get_text() for text in legend.get_texts()] == [
    "label=lightgbm-logit, normalize=5",
    "label=tabpfn-quantiles-logit, normalize=5",
  ]
