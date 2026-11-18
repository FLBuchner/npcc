"""Plotting helpers for the simulation study (matplotlib only).

These consume the tidy metric table :func:`npcc.experiments.run_study` returns:
pick a ``(family, tau_scenario, quantity, metric)`` slice and compare the
estimators in it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch


def _slice(
  df: pd.DataFrame, family: str, tau_scenario: str, quantity: str
) -> pd.DataFrame:
  sliced = df[
    (df["family"] == family)
    & (df["tau_scenario"] == tau_scenario)
    & (df["quantity"] == quantity)
  ]
  return cast("pd.DataFrame", sliced)


def metric_boxplot_by_n(
  metrics_df: pd.DataFrame,
  *,
  family: str,
  tau_scenario: str,
  quantity: str,
  metric: str,
  label: str | Sequence[str] | None = None,
  normalize: str | Sequence[str] | None = None,
  ax: Axes | None = None,
) -> Axes:
  """Plot per-repetition metric boxplots by sample size and model identity.

  For conditional scenarios, metrics are averaged over conditioning values
  within each repetition before plotting. Each model identity is the
  ``(label, normalize)`` combination (``label`` already encodes the backend,
  transform, and hyperparameters). A selector may be ``None`` to include all
  values, a string to select one value, or a sequence of strings to select
  multiple values.
  """
  ax = ax or plt.subplots(figsize=(6.0, 3.5))[1]
  sub = _slice(metrics_df, family, tau_scenario, quantity).dropna(
    subset=[metric]
  )
  model_columns = ["label", "normalize"]
  selectors = {
    "label": label,
    "normalize": normalize,
  }
  for column, selector in selectors.items():
    if selector is None:
      continue
    values = [selector] if isinstance(selector, str) else list(selector)
    sub = sub[sub[column].isin(values)]

  per_rep = sub.groupby(
    ["n", *model_columns, "rep"], as_index=False, dropna=False
  )[metric].mean()
  sample_sizes = sorted(per_rep["n"].unique())
  models = sorted(
    per_rep[model_columns].drop_duplicates().itertuples(index=False, name=None),
    key=lambda values: tuple(str(value) for value in values),
  )

  n_models = len(models)
  group_width = 0.8
  box_width = group_width / max(n_models, 1)
  color_map = plt.get_cmap("tab10")
  handles: list[Patch] = []
  for model_index, model_values in enumerate(models):
    offset = (model_index - (n_models - 1) / 2) * box_width
    positions = [index + offset for index in range(len(sample_sizes))]
    model_rows = per_rep
    for column, value in zip(model_columns, model_values, strict=True):
      model_rows = model_rows[model_rows[column] == value]
    data = [
      model_rows[model_rows["n"] == n][metric].to_numpy() for n in sample_sizes
    ]
    color = color_map(model_index % 10)
    boxes = ax.boxplot(
      data,
      positions=positions,
      widths=box_width * 0.9,
      patch_artist=True,
      manage_ticks=False,
    )
    for box in boxes["boxes"]:
      box.set_facecolor(color)
    label = ", ".join(
      f"{column}={value}"
      for column, value in zip(model_columns, model_values, strict=True)
    )
    handles.append(Patch(facecolor=color, label=label))

  ax.set_xticks(range(len(sample_sizes)), [str(n) for n in sample_sizes])
  ax.set_xlabel("Sample size")
  ax.set_ylabel(metric)
  ax.set_title(f"{family} / {tau_scenario} / {quantity}")
  if handles:
    ax.legend(
      handles=handles,
      title="Model",
      fontsize="small",
      loc="center left",
      bbox_to_anchor=(1.02, 0.5),
    )
  return ax
