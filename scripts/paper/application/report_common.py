"""What the application's report scripts share: a run's tables, and pages.

Every report reads a finished run folder (``npcc-application evaluate`` has
written its ``tables/``) and fits nothing. Pages are A4 landscape figures
saved into one PDF, numbered as they are added.
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

PAGE = (11.7, 8.3)
ROWS_PER_PAGE = 24


def parser(description: str) -> argparse.ArgumentParser:
  """The options every report takes: the run folder and where to write."""
  p = argparse.ArgumentParser(description=description)
  p.add_argument("--run", required=True, type=Path, help="The run folder.")
  p.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Where to write; <run>/reports by default.",
  )
  return p


def read_table(run: Path, name: str) -> pd.DataFrame:
  """One of ``evaluate``'s tables, refusing a run it has not evaluated."""
  path = Path(run) / "tables" / f"{name}.csv"
  if not path.exists():
    raise FileNotFoundError(
      f"{path} must exist; run `npcc-application evaluate` on {run} first"
    )
  return pd.read_csv(path)


def number(value: object) -> str:
  """Three significant digits, or N/A."""
  try:
    v = float(value)  # ty: ignore[invalid-argument-type] - table cells are numbers or strings
  except (TypeError, ValueError):
    return str(value)
  return f"{v:.3g}" if np.isfinite(v) else "N/A"


def with_interval(frame: pd.DataFrame, metric: str) -> pd.Series:
  """``mean [low, high]`` per row, from ``metric``, ``_low`` and ``_high``."""
  return frame.apply(
    lambda r: (
      f"{number(r[metric])} [{number(r[metric + '_low'])}, "
      f"{number(r[metric + '_high'])}]"
    ),
    axis=1,
  )


class Report:
  """A PDF that numbers its pages, written whole or not at all."""

  def __init__(self, path: Path, title: str) -> None:
    self.path = path
    self.title = title
    self.pages = 0

  def __enter__(self) -> Self:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.partial = self.path.with_name(f".{self.path.name}.partial")
    self.pdf = PdfPages(self.partial)
    self.pdf.infodict()["Title"] = self.title
    return self

  def __exit__(self, kind: object, *_: object) -> None:
    self.pdf.close()
    if kind is None:
      self.partial.replace(self.path)
    else:
      self.partial.unlink(missing_ok=True)

  def save(self, figure: Figure) -> None:
    """Add a page and close its figure."""
    self.pages += 1
    figure.text(0.99, 0.006, str(self.pages), ha="right", fontsize=8)
    self.pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)

  def text(self, title: str, lines: Sequence[str]) -> None:
    """A page of prose."""
    figure = plt.figure(figsize=PAGE)
    figure.text(0.06, 0.9, title, fontsize=18, weight="bold")
    y = 0.8
    for line in lines:
      figure.text(0.06, y, line, fontsize=11, va="top", wrap=True)
      y -= 0.07
    self.save(figure)

  def table(self, frame: pd.DataFrame, title: str, note: str = "") -> None:
    """A table over as many pages as it needs, numbers to 3 digits."""
    if frame.empty:
      self.text(title, ["No rows.", note])
      return
    for start in range(0, len(frame), ROWS_PER_PAGE):
      part = frame.iloc[start : start + ROWS_PER_PAGE].copy()
      for column in part:
        if pd.api.types.is_numeric_dtype(part[column]):
          part[column] = part[column].map(number)
      figure, axis = plt.subplots(figsize=PAGE)
      axis.axis("off")
      suffix = (
        f" (rows {start + 1}-{start + len(part)})"
        if len(frame) > ROWS_PER_PAGE
        else ""
      )
      figure.suptitle(title + suffix, fontsize=13)
      table = axis.table(
        cellText=part.to_numpy(),
        colLabels=list(part.columns),
        cellLoc="center",
        loc="center",
      )
      table.auto_set_font_size(False)
      table.set_fontsize(7)
      table.scale(1, 1.4)
      table.auto_set_column_width(list(range(len(part.columns))))
      if note:
        figure.text(0.02, 0.02, note, fontsize=8, wrap=True)
      self.save(figure)


def _horizontal(axis: Axes) -> dict[str, object]:
  """Horizontal boxes, in the keyword the installed matplotlib takes."""
  if "orientation" in inspect.signature(axis.boxplot).parameters:
    return {"orientation": "horizontal"}
  return {"vert": False}


def boxes(
  axis: Axes,
  frame: pd.DataFrame,
  metric: str,
  groups: Sequence[str],
  *,
  by: str,
  title: str,
  zero_line: bool = False,
) -> None:
  """Horizontal boxes of ``metric`` per group: quartiles, 1.5 IQR, mean."""
  data = [
    frame.loc[frame[by] == g, metric].dropna().to_numpy(dtype=float)
    for g in groups
  ]
  kept = [(i, d) for i, d in enumerate(data) if len(d)]
  if kept:
    axis.boxplot(
      [d for _, d in kept],
      positions=[i for i, _ in kept],
      **_horizontal(axis),
      showfliers=False,
      showmeans=True,
      meanprops={
        "marker": "D",
        "markerfacecolor": "darkorange",
        "markeredgecolor": "darkorange",
        "markersize": 3,
      },
    )
  axis.set(yticks=range(len(groups)), yticklabels=list(groups), title=title)
  axis.set_ylim(len(groups) - 0.5, -0.5)
  axis.tick_params(labelsize=7)
  axis.grid(axis="x", alpha=0.2)
  if zero_line:
    axis.axvline(0.0, color="black", linewidth=0.6)


def box_page(
  report: Report,
  frame: pd.DataFrame,
  metrics: Sequence[tuple[str, str, bool]],
  groups: Sequence[str],
  *,
  by: str,
  title: str,
  note: str,
) -> None:
  """A grid of :func:`boxes`, one per ``(metric, label, zero_line)``."""
  columns = 3 if len(metrics) > 4 else 2
  rows = int(np.ceil(len(metrics) / columns))
  figure, axes = plt.subplots(
    rows, columns, figsize=PAGE, squeeze=False, layout="constrained"
  )
  for axis, (metric, label, zero) in zip(axes.flat, metrics, strict=False):
    boxes(axis, frame, metric, groups, by=by, title=label, zero_line=zero)
  for axis in axes.flat[len(metrics) :]:
    axis.set_visible(False)
  figure.suptitle(title, fontsize=13)
  figure.text(0.01, -0.03, note, fontsize=8)
  report.save(figure)


def timing_pages(
  report: Report,
  summary: pd.DataFrame,
  per_station: pd.DataFrame,
  *,
  by: str,
  title: str,
) -> None:
  """The stage timings: a table over stations, and a figure of them.

  ``summary`` is ``evaluate``'s timing table; ``per_station`` holds
  ``station``, ``by``, ``stage`` and ``seconds``.
  """
  note = (
    "Seconds per stage, summed over a station's responses, then summarized "
    "over stations (each weighted equally). Worker time, not wall time."
  )
  report.table(summary, title, note)
  if per_station.empty:
    return
  groups = list(dict.fromkeys(per_station[by]))
  figure, (left, right) = plt.subplots(1, 2, figsize=PAGE, layout="constrained")
  means = per_station.groupby([by, "stage"])["seconds"].mean().unstack()
  means = means.reindex(groups).fillna(0.0)
  base = np.zeros(len(groups))
  colors = plt.get_cmap("tab20")
  for k, stage in enumerate(means.columns):
    values = means[stage].to_numpy()
    left.barh(
      range(len(groups)), values, left=base, label=stage, color=colors(k % 20)
    )
    base += values
  left.set(
    yticks=range(len(groups)),
    yticklabels=groups,
    xlabel="Mean seconds per station",
    title="By stage",
  )
  left.set_ylim(len(groups) - 0.5, -0.5)
  left.legend(fontsize=7)
  totals = per_station.groupby(["station", by])["seconds"].sum().reset_index()
  boxes(right, totals, "seconds", groups, by=by, title="Station totals (s)")
  figure.suptitle(title, fontsize=13)
  report.save(figure)


def pit_page(
  report: Report,
  cases: pd.DataFrame,
  models: Sequence[str],
  *,
  title: str,
  bins: int = 10,
) -> None:
  """Held-out PIT histograms, one panel per model, pooled over ``cases``."""
  columns = 2
  rows = int(np.ceil(len(models) / columns))
  figure, axes = plt.subplots(
    rows, columns, figsize=PAGE, squeeze=False, layout="constrained"
  )
  edges = np.linspace(0.0, 1.0, bins + 1)
  for axis, model in zip(axes.flat, models, strict=False):
    pit = cases.loc[cases["model"] == model, "pit"].to_numpy(dtype=float)
    if len(pit):
      axis.hist(
        pit, bins=edges, density=True, color="steelblue", edgecolor="white"
      )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlim=(0, 1), xlabel="PIT", ylabel="Density", title=model)
    axis.text(
      0.98,
      0.95,
      f"n = {len(pit):,}",
      transform=axis.transAxes,
      ha="right",
      va="top",
      fontsize=8,
    )
  for axis in axes.flat[len(models) :]:
    axis.set_visible(False)
  figure.suptitle(title + "\nDashed: the uniform density", fontsize=12)
  report.save(figure)


def ordered(values: pd.Series, first: Callable[[str], bool]) -> list[str]:
  """The distinct values in order of appearance, those ``first`` picks first."""
  seen = list(dict.fromkeys(str(v) for v in values))
  return [v for v in seen if first(v)] + [v for v in seen if not first(v)]
