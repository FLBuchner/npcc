"""The import surface: which layer may reach which, and what `import npcc` costs.

Two invariants, both of which the package documents and neither of which
anything checked before.

``npcc.core.registry``'s module docstring promises that "``import npcc`` and
:func:`available_backends` never import NGBoost / TabICL / LightGBM",
which is what makes the optional extras optional. Nothing enforced it, and one
eager import anywhere in the chain -- a convenience re-export, a type
annotation moved out from behind ``TYPE_CHECKING`` -- would turn an extra into
a hard dependency without failing anything.

And the layers: ``npcc.experiments`` is behind its own extra and sits above
``npcc.core``, so the edge points one way. A stray import the other way would
put pandas and matplotlib on the path of every ``import npcc``.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "npcc"

#: Third-party packages reachable **only** through an optional extra.
#: Importing `npcc` must not pull in any of them.
#:
#: `matplotlib`, `pandas` and `sklearn` are absent on purpose, and it is worth
#: saying why: each is also a transitive hard dependency -- `matplotlib` of
#: `pyvinecopulib`, `pandas` and `sklearn` of `tabpfn` -- so `import npcc`
#: loads them whatever this package does, and asserting otherwise would pin a
#: fact about someone else's dependency tree.
_OPTIONAL = (
  "catboost",
  "lightgbm",
  "ngboost",
  "pytabkit",
  "synthefy_nori",
  "tabicl",
  "xgboost",
)


def _imported_by(statements: str) -> set[str]:
  """Top-level third-party modules present after running ``statements``.

  A fresh interpreter, because `sys.modules` in this one already holds
  everything the test suite imported.

  Parameters
  ----------
  statements : str
      Python source to run before the check.

  Returns
  -------
  set of str
      The names from ``_OPTIONAL`` that ended up imported.
  """
  probe = (
    f"{statements}\n"
    "import sys\n"
    f"present = [n for n in {_OPTIONAL!r} if n in sys.modules]\n"
    "print(' '.join(present))\n"
  )
  result = subprocess.run(
    [sys.executable, "-c", probe],
    capture_output=True,
    text=True,
    check=True,
    cwd=_ROOT,
  )
  return set(result.stdout.split())


def test_importing_npcc_needs_no_optional_extra() -> None:
  assert _imported_by("import npcc") == set()


def test_listing_the_backends_needs_no_optional_extra() -> None:
  """Naming what is available must not construct any of it."""
  assert _imported_by("import npcc; npcc.available_backends()") == set()


def test_validating_backend_kwargs_needs_no_optional_extra() -> None:
  """Building fit controls validates the backend name, and stops there."""
  assert (
    _imported_by(
      "from npcc import FitControlsRosenblattBicop\n"
      "FitControlsRosenblattBicop(backend='ngboost')"
    )
    == set()
  )


def _imports(path: Path, *, module_scope: bool = False) -> set[str]:
  """Every module named by an import in ``path``, dotted.

  Parameters
  ----------
  path : pathlib.Path
      The Python file to read.
  module_scope : bool, default=False
      Whether to leave out the imports inside a function body, which run only
      when the function is called.

  Returns
  -------
  set of str
      Full dotted module names, so a layer prefix or one module can be
      matched; ``name.split(".")[0]`` is the package.
  """
  tree = ast.parse(path.read_text())
  nested = (
    {
      node
      for parent in ast.walk(tree)
      if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
      for node in ast.walk(parent)
    }
    if module_scope
    else set()
  )
  names: set[str] = set()
  for node in ast.walk(tree):
    if node in nested:
      continue
    if isinstance(node, ast.Import):
      names.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
      names.add(node.module)
  return names


def _needs_extra(path: Path) -> bool:
  """Whether importing ``path`` imports an optional extra, through adapters.

  Parameters
  ----------
  path : pathlib.Path
      An adapter module under ``npcc/core/backends``.

  Returns
  -------
  bool
      ``True`` if it, or an adapter it imports at module scope, imports a
      package from ``_OPTIONAL``.
  """
  imported = _imports(path, module_scope=True)
  if {m.split(".")[0] for m in imported} & set(_OPTIONAL):
    return True
  prefix = "npcc.core.backends."
  return any(
    _needs_extra(path.parent / f"{m.removeprefix(prefix)}.py")
    for m in imported
    if m.startswith(prefix)
  )


def test_the_registry_imports_no_backend_module_at_module_scope() -> None:
  """The deferral that makes the extras optional lives in the factories.

  Each adapter module imports its own third-party package at module scope --
  which is right, the module being that package's adapter -- so what keeps
  `import npcc` cheap is entirely that the registry reaches those modules only
  from inside a factory body. One module-scope import here promotes every
  extra to a hard dependency at once.
  """
  module_scope = _imports(_SRC / "core" / "registry.py", module_scope=True)

  assert {
    m for m in module_scope if m.startswith("npcc.core.backends")
  } == set()


@pytest.mark.parametrize(
  "path",
  sorted((_SRC / "core" / "backends").glob("*.py")),
  ids=lambda p: p.name,
)
def test_every_adapter_is_reached_only_through_a_factory(path: Path) -> None:
  """An adapter that needs an extra is imported eagerly only by adapters.

  Another adapter may: it is itself reached only through a factory, which its
  own case here checks, since it needs the extra too -- through the adapter it
  subclasses, if not directly. ``npcc/__init__.py`` re-exports the TabPFN
  adapter, which passes because `tabpfn` is a hard dependency, not an extra.
  """
  backends = _SRC / "core" / "backends"
  if path.name == "__init__.py":
    pytest.skip("no third-party import of its own")
  if not _needs_extra(path):
    pytest.skip("needs no optional extra")

  module = f"npcc.core.backends.{path.stem}"
  eager = [
    other
    for other in sorted(_SRC.rglob("*.py"))
    if other.parent != backends and module in _imports(other, module_scope=True)
  ]

  assert eager == [], f"{module} imported at module scope by {eager}"


@pytest.mark.parametrize(
  "path",
  sorted((_SRC / "core").rglob("*.py")),
  ids=lambda p: str(p.relative_to(_SRC)),
)
def test_core_never_imports_the_layer_above_it(path: Path) -> None:
  """``npcc.experiments`` is above ``npcc.core``; the edge points one way.

  Read from the AST rather than as a substring: a docstring or comment naming
  the layer above is documentation -- a branch in ``core`` may exist because of
  a data shape the experiments package produces, and saying so is useful --
  while an ``import`` is the dependency this forbids.
  """
  imported = {
    name
    for name in _imports(path)
    if name == "npcc.experiments" or name.startswith("npcc.experiments.")
  }

  assert imported == set(), (
    f"{path.relative_to(_ROOT)} imports {sorted(imported)}"
  )


def test_every_core_name_is_also_reachable_from_the_top() -> None:
  """The two public surfaces agree, so either spelling works.

  ``npcc`` carries more -- the concrete backend leaves, which live in
  ``npcc.core.backends`` rather than in ``npcc.core`` itself.
  """
  import npcc
  import npcc.core

  assert set(npcc.core.__all__) <= set(npcc.__all__)
  assert set(npcc.__all__) - set(npcc.core.__all__) == {
    "TabPFNQuantileBackend",
  }
  for name in npcc.core.__all__:
    assert getattr(npcc, name) is getattr(npcc.core, name), name


@pytest.mark.parametrize("module", ["npcc", "npcc.core"])
def test_the_public_lists_are_sorted_and_resolve(module: str) -> None:
  """A name in ``__all__`` that does not resolve is a broken re-export."""
  import importlib

  imported = importlib.import_module(module)
  names = list(imported.__all__)

  assert names == sorted(names)
  for name in names:
    assert hasattr(imported, name), name


#: Where an upstream name can be mentioned: code, and the two prose files that
#: name symbols. `pyproject.toml` is excluded because it holds URLs rather than
#: references -- `.../pyvinecopulib.git` is not a symbol.
_PROSE = ("AGENTS.md", "README.md")

#: A dotted `pyvinecopulib.X.Y`, not preceded by `/` so a URL path does not
#: match.
_UPSTREAM_REF = re.compile(
  r"(?<![/\w])pyvinecopulib(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)


def _upstream_references() -> dict[str, list[str]]:
  """Every dotted ``pyvinecopulib`` name this package mentions, and where.

  Returns
  -------
  dict
      Dotted name to the files mentioning it, relative to the repository
      root. Empty only if the globbing below is broken, which the caller
      checks -- see the floor in
      :func:`test_every_upstream_name_this_package_mentions_resolves`.
  """
  paths = [
    *sorted(_SRC.rglob("*.py")),
    *sorted((_ROOT / "tests").rglob("*.py")),
    *[_ROOT / name for name in _PROSE],
  ]
  found: dict[str, list[str]] = {}
  for path in paths:
    # This file spells a placeholder to describe the pattern, so it cannot be
    # its own subject -- the same reason `test_prose.py` exempts itself.
    if not path.is_file() or path.name == "test_import_surface.py":
      continue
    for match in _UPSTREAM_REF.finditer(path.read_text()):
      found.setdefault(match.group(0), []).append(str(path.relative_to(_ROOT)))
  return found


def _resolves(dotted: str) -> bool:
  """Whether ``dotted`` names something that exists at the pinned version.

  Splits at every boundary rather than guessing where the module ends, since
  a dotted name's module prefix and its attribute tail are not fixed: the
  pipeline steps live one level deeper than the contract classes.

  Parameters
  ----------
  dotted : str
      A dotted name beginning ``pyvinecopulib``.

  Returns
  -------
  bool
      ``True`` if the name resolves.
  """
  parts = dotted.split(".")
  for split in range(len(parts), 0, -1):
    try:
      obj = importlib.import_module(".".join(parts[:split]))
    except ImportError:
      continue
    for attr in parts[split:]:
      obj = getattr(obj, attr, None)
      if obj is None:
        return False
    return True
  return False


def test_every_upstream_name_this_package_mentions_resolves() -> None:
  """A cross-reference to a moved or renamed upstream name is a lying doc.

  Prose is the half that nothing else checks: a stale ``:func:`` role fails no
  import and no type check, and this package tracks a pre-release pin whose
  surface has moved twice -- the input-pipeline steps went from private
  modules to ``core`` and then to ``core.extend``, and each move left
  references behind in docstrings that only a sweep like this finds.

  Written after an upstream author reported the same shape three times in one
  day: a hand-written list applied across a set, right in nine places out of
  ten. Re-reading does not catch the tenth; cross-checking the set does.

  The floor below is not decoration. "Nothing unresolved" is also what a
  broken glob reports, and a check whose failure mode is silent success is
  the fault it exists to prevent -- so each of the three roots has to have
  contributed at least one reference for the empty result to mean anything.
  """
  references = _upstream_references()
  roots = {
    where.split("/")[0] for places in references.values() for where in places
  }

  assert {"src", "tests"} <= roots, roots
  assert any(
    place.endswith(".md") for places in references.values() for place in places
  )

  unresolved = {
    name: sorted(set(where))
    for name, where in sorted(references.items())
    if not _resolves(name)
  }

  assert unresolved == {}, unresolved
