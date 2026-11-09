"""The quantile-table read-out every NPCC row of the paper must come from.

The read-out changed at ``SINCE``: one weighted smoother for every backend,
each distinct conditioning row predicted once, and the normal-grid levels the
paper reports. A row an earlier commit produced used the old read-out, so a
collector that met one beside the new rows would average two estimators
under one name. The collectors call :func:`check` on every NPCC row they
read; competitors that never touch the read-out (VDC, TLL, the parametric
fits, the R scripts) are not checked.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Iterable

#: The first commit with the current read-out.
SINCE = "7dbd778"
REPO = Path(__file__).resolve().parents[2]


@functools.cache
def descends(commit: str) -> bool:
  """Whether ``commit`` is ``SINCE`` or one of its descendants."""
  return (
    subprocess.run(
      ["git", "-C", str(REPO), "merge-base", "--is-ancestor", SINCE, commit],
      capture_output=True,
      check=False,
    ).returncode
    == 0
  )


def check(commits: Iterable[object], what: str) -> None:
  """Exit unless every commit in ``commits`` is ``SINCE`` or descends from it.

  Parameters
  ----------
  commits : iterable
      The ``npcc_commit`` of every NPCC row read.
  what : str
      The rows, for the message.
  """
  stale = sorted({str(c) for c in commits if not descends(str(c))})
  if stale:
    raise SystemExit(
      f"{what} must come from the read-out of {SINCE} or a later commit; got "
      f"rows from {stale}. Move the old outputs under results/superseded_*/ "
      "or rerun them."
    )
