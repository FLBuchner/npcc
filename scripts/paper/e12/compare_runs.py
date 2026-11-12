"""E12: compare two runs' CSVs row by row, every value as it was written.

Each CSV is read as text, so two runs agree on a value only if they wrote
it to the same digits. Rows are matched on the file's keys (``KEYS``). Per
file in both runs, the report gives the rows that agree in every shared
column, the rows that differ with the columns that do, the rows only one
run has (counted per design and d, and per study and arm where the file has
them), and the columns only one run has.

Use it to check a rerun of E12 against an earlier one: once the studies
still being written have finished, a rerun on the full folders differs from
this run only in the cells those studies added.

**Usage**::

    python scripts/paper/e12/compare_runs.py <earlier run> <later run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: Each E12 CSV and the columns that name its rows.
KEYS = {
  "mass_e3_cells.csv": ["study", "arm", "design", "d", "seed"],
  "mass_e10_members.csv": ["study", "arm", "design", "d", "seed", "member"],
  "mass_e3ac_cells.csv": ["study", "arm", "design", "d", "n_train", "seed"],
  "mass_e3_summary.csv": ["study", "design", "d", "arm_s"],
  "final_table_e3d_e3e_e10.csv": ["design", "d", "arm_s"],
  "final_table_e3c.csv": ["design", "d", "arm"],
  "final_table_e3a.csv": ["arm", "n_train"],
  "mass_e3_cells_reported.csv": ["study", "arm", "design", "d", "seed"],
  "mass_e3ac_cells_reported.csv": [
    "study",
    "arm",
    "design",
    "d",
    "n_train",
    "seed",
  ],
  "mass_paired.csv": ["comp", "design", "d"],
  "mass_vdc_pairs.csv": ["comp", "design", "d"],
}

#: The columns to count rows by when one run lacks them, as present.
GROUPS = ["study", "arm", "arm_s", "comp", "design", "d", "n_train"]


def compare(name: str, a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
  """The report on one file, read as text from runs ``a`` and ``b``.

  Parameters
  ----------
  name : str
      The file's name, a key of ``KEYS``.
  a, b : pd.DataFrame
      Its rows in each run, every value a string.

  Returns
  -------
  list of str
      The report's lines.
  """
  keys = KEYS[name]
  shared = [c for c in a.columns if c in b.columns and c not in keys]
  lines = []
  only_cols_a = [c for c in a.columns if c not in b.columns]
  only_cols_b = [c for c in b.columns if c not in a.columns]
  if only_cols_a or only_cols_b:
    lines.append(
      f"  columns only in A: {only_cols_a}; only in B: {only_cols_b}"
    )
  m = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
  both = m[m["_merge"] == "both"]
  differ = []
  for _, row in both.iterrows():
    cols = [c for c in shared if row[f"{c}_a"] != row[f"{c}_b"]]
    if cols:
      differ.append((tuple(row[k] for k in keys), cols))
  lines.append(
    f"  rows in both: {len(both)}; equal in every shared column: "
    f"{len(both) - len(differ)}; differing: {len(differ)}"
  )
  for key, cols in differ:
    lines.append(f"    differs {dict(zip(keys, key, strict=True))}: {cols}")
  for side, label in (("left_only", "only in A"), ("right_only", "only in B")):
    rows = m[m["_merge"] == side]
    if len(rows):
      by = [c for c in GROUPS if c in keys]
      counts = rows.groupby(by).size()
      lines.append(f"  rows {label}: {len(rows)}")
      lines += [
        f"    {dict(zip(by, k if isinstance(k, tuple) else (k,), strict=True))}"
        f": {n}"
        for k, n in counts.items()
      ]
  return lines


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("a", type=Path, help="run A")
  p.add_argument("b", type=Path, help="run B")
  a = p.parse_args()
  print(f"A = {a.a}\nB = {a.b}")
  for name in KEYS:
    fa, fb = a.a / name, a.b / name
    if not (fa.is_file() and fb.is_file()):
      print(f"{name}: in A {fa.is_file()}, in B {fb.is_file()}; not compared")
      continue
    print(f"{name}:")
    left = pd.read_csv(fa, dtype=str, keep_default_na=False)
    right = pd.read_csv(fb, dtype=str, keep_default_na=False)
    print("\n".join(compare(name, left, right)))


if __name__ == "__main__":
  main()
