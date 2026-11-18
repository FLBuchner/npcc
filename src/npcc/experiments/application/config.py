"""The application's configuration, read from one TOML file.

It names the data and which of its cases enter (``[data]``), the marginal
models and how each response is put on its model's scale (``[margins]``,
``[responses]``), the copula arms and their Monte Carlo settings
(``[copulas]``), where each backend runs (``[devices]``), and the npcc
estimators (``[[estimators]]``), which are the simulation study's own entries
copied verbatim, so that a label means the same backend configuration in both.

A marginal model is ``emos-seasonal`` or ``<backend>-<scope>``: an
estimator's backend on the ``all`` or the ``emos`` covariates. Its transform
is the response's, not the estimator's, since each response has its own
support; the estimator contributes its backend and ``backend_kwargs``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast, get_args

import numpy as np
import torch

from npcc.core.controls import Transform
from npcc.core.errors import EstimatorConfigError
from npcc.experiments.application.data import (
  DATA_SHA256,
  DEFAULT_DATA_DIR,
  CaseSelection,
)
from npcc.experiments.application.derived import (
  DERIVATIONS,
  METEOSWISS,
  Derivation,
)
from npcc.experiments.config import EstimatorSpec, _parse_estimator

#: The seasonal EMOS baseline's model name.
EMOS_MODEL = "emos-seasonal"
#: The covariate scopes a ``<backend>-<scope>`` marginal model may name.
MARGIN_SCOPES: tuple[str, ...] = ("all", "emos")
#: Which training PITs the copulas are fitted to: the margin fitted on every
#: training case evaluated at them (``in_sample``), or each case's PIT from a
#: margin fitted without its initialization year (``out_of_fold``).
TRAIN_PITS: tuple[str, ...] = ("in_sample", "out_of_fold")
#: The dtypes a margin's test quantiles may be stored in. Under ``float32``
#: they are rounded to it as soon as they are predicted, before anything
#: scores or stores them, so the copula stage, which reads them from the
#: files, sees exactly what the margin stage scored.
QUANTILE_DTYPES: tuple[str, ...] = ("float64", "float32")
#: What separates a copula arm from its design, as in ``gaussian@5d``: an arm
#: without it runs on ``[responses] modeled`` with ``derived`` drawn from it,
#: one with it on the responses ``[copulas.designs.<name>]`` names.
ARM_DESIGN: str = "@"
_DESIGN_NAME = re.compile(r"[a-z0-9_]+")


def split_arm(arm: str) -> tuple[str, str | None]:
  """An arm's base name and its design, ``None`` for the main one.

  Parameters
  ----------
  arm : str
      A copula arm as configured, such as ``gaussian`` or ``gaussian@5d``.

  Returns
  -------
  base : str
      The arm the copula stage fits, such as ``gaussian``.
  design : str or None
      The design's name, such as ``5d``; ``None`` without one.

  Raises
  ------
  ValueError
      If the name has an empty part or more than one separator.
  """
  base, sep, design = arm.partition(ARM_DESIGN)
  if not sep:
    return arm, None
  if not base or not _DESIGN_NAME.fullmatch(design):
    raise ValueError(
      f"a copula arm must be '<arm>' or '<arm>{ARM_DESIGN}<design>', the "
      f"design of lowercase letters, digits and underscores; got {arm!r}"
    )
  return base, design


@dataclass(frozen=True)
class CopulaDesign:
  """Which responses a copula arm models, and which it derives from them.

  Attributes
  ----------
  modeled : tuple of str
      The responses the copula is fitted to, sorted as the data sorts them.
  derived : tuple of str
      The responses computed from each of its draws
      (:mod:`~npcc.experiments.application.derived`), sorted likewise.
  """

  modeled: tuple[str, ...]
  derived: tuple[str, ...] = ()


_SUPPORTS: dict[str, tuple[float, float]] = {
  "identity": (-math.inf, math.inf),
  "log": (0.0, math.inf),
  "logit": (0.0, 1.0),
  "probit": (0.0, 1.0),
}


@dataclass(frozen=True)
class ResponseSpec:
  """How one response is put on its marginal model's scale.

  The model sees ``y / scale``. On the unit interval, an observation exactly
  at either bound is first moved ``boundary_nudge`` inside it, in the
  response's own units, since no continuous model there gives a bound
  positive density; on ``(0, inf)``, one at or below 0 is refused.

  Attributes
  ----------
  transform : str
      The npcc margin's response transform; its support is the model scale's.
  scale : float, default=1.0
      The response's unit relative to the model scale's, such as 100 for a
      percentage modeled as a fraction.
  boundary_nudge : float, default=0.0
      Distance an observation at a bound of the unit interval is moved
      inside it, in the response's units. Positive for ``logit`` and
      ``probit``, zero otherwise.
  """

  transform: Transform
  scale: float = 1.0
  boundary_nudge: float = 0.0

  def __post_init__(self) -> None:
    """Refuse a transform, scale or nudge that cannot be applied."""
    if self.transform not in _SUPPORTS:
      raise ValueError(
        f"transform must be one of {list(_SUPPORTS)}; got {self.transform!r}"
      )
    if not (math.isfinite(self.scale) and self.scale > 0.0):
      raise ValueError(f"scale must be positive and finite; got {self.scale}")
    if self.support == (0.0, 1.0):
      if not 0.0 < self.boundary_nudge < 0.5 * self.scale:
        raise ValueError(
          f"boundary_nudge must lie in (0, scale / 2) for transform "
          f"{self.transform!r}, whose bounds an observation can sit on; got "
          f"{self.boundary_nudge} with scale {self.scale}"
        )
    elif self.boundary_nudge != 0.0:
      raise ValueError(
        f"boundary_nudge must be 0 for transform {self.transform!r}, which "
        f"has no upper bound to move an observation off; got "
        f"{self.boundary_nudge}"
      )

  @property
  def support(self) -> tuple[float, float]:
    """The model scale's support, ``(lo, hi)``."""
    return _SUPPORTS[self.transform]

  @property
  def log_scale(self) -> float:
    """``log(scale)``: what a model-scale log-density loses on the way back."""
    return math.log(self.scale)

  def to_model(self, y: np.ndarray, name: str) -> np.ndarray:
    """Responses on the model scale.

    Parameters
    ----------
    y : numpy.ndarray
        Observations in the response's own units.
    name : str
        The response, for the message.

    Returns
    -------
    numpy.ndarray
        A new array, ``y / scale`` after the bound handling above.
    """
    out = np.array(y, dtype=np.float64, copy=True)
    lo, hi = self.support
    if hi == 1.0:
      outside = (out < 0.0) | (out > self.scale)
      if outside.any():
        raise ValueError(
          f"{name} must lie in [0, {self.scale}] for transform "
          f"{self.transform!r}; got {int(outside.sum())} values outside"
        )
      out[out == 0.0] = self.boundary_nudge
      out[out == self.scale] = self.scale - self.boundary_nudge
    elif lo == 0.0:
      bad = out <= 0.0
      if bad.any():
        raise ValueError(
          f"{name} must be positive for transform {self.transform!r}; got "
          f"{int(bad.sum())} values at or below 0"
        )
    return out / self.scale


@dataclass(frozen=True)
class MarginalModel:
  """A marginal model, parsed from its name.

  Attributes
  ----------
  name : str
      ``emos-seasonal`` or ``<backend>-<scope>``.
  backend : str or None
      The npcc backend; ``None`` for EMOS.
  scope : str
      The covariate scope it is fitted on.
  """

  name: str
  backend: str | None
  scope: str

  @property
  def is_emos(self) -> bool:
    """Whether this is the EMOS baseline."""
    return self.backend is None


@dataclass(frozen=True)
class ApplicationConfig:
  """Everything that defines an application run but the command line.

  Attributes
  ----------
  data_dir : pathlib.Path
      The folder holding ``x.nc`` and ``y.nc``: relative to the working
      directory unless absolute, ``~`` expanded.
  stations : tuple of int or None
      The stations to run; ``None`` for every station in the data.
  selection : CaseSelection
      The lead, initialization hour and split dates.
  models : tuple of str
      The marginal models, in run order.
  responses : mapping of str to ResponseSpec
      Every response's scale, by its name in ``y.nc``.
  estimators : tuple of EstimatorSpec
      The npcc estimators, at most one per backend.
  copula_arms : tuple of str
      The copula arms, checked against the copula stage's list when it runs.
  copula_margins : tuple of str or None
      The marginal models whose sets enter the copula stage, drawn from
      ``models``; ``None`` for all of them (see :attr:`copula_models`).
  copula_arms_by_margin : mapping of str to tuple of str
      For a copula stage's marginal model named here, the only arms it runs,
      drawn from ``copula_arms``; a model not named runs every arm (see
      :meth:`copula_arms_for`). Empty by default.
  verify_data : bool
      Whether to refuse input files that are not the published ones, by
      their SHA-256; off only for files of another source, such as
      synthetic ones.
  modeled_responses : tuple of str or None
      The responses the margins and copulas are fitted to; ``None`` for
      every response (see :attr:`modeled`).
  derived_responses : tuple of str
      The responses computed from each copula draw of the modeled ones
      (:mod:`~npcc.experiments.application.derived`) and scored with them;
      none by default.
  magnus : str
      The Magnus formula of the derivations, a key of
      :data:`~npcc.experiments.application.derived.MAGNUS`.
  devices : mapping of str to str
      The device of each backend named; an unnamed one picks its own.
  n_levels : int
      The number of quantile levels, at ``(k + 0.5) / n_levels``.
  n_draws : int
      Copula draws per test case.
  sample_batch_size : int
      The most rows an npcc copula samples in one call.
  mixture_k : int
      Members of a mixture copula arm.
  base_seed : int
      The seed every fit's seed is derived from.
  batch_size : int or None
      The npcc backends' inference batch size; ``None`` for their default.
  train_pits : str
      One of :data:`TRAIN_PITS`: the training PITs the copulas are fitted
      to. ``out_of_fold`` makes the margin stage also fit each marginal
      model once per training year without that year's cases and store
      their PITs beside the in-sample ones.
  save_draws : bool
      Whether the copula stage stores each arm's draws of every response,
      per test case, beside its scores; a mixture's members are scored on
      their own draws, but only the mixture's are stored.
  copula_designs : mapping of str to CopulaDesign
      The copula stage's other designs, by name: an arm named
      ``<arm>@<name>`` runs on that design's responses rather than on
      ``modeled`` and ``derived``. The margins are fitted to every response
      any design models (:attr:`margin_responses`). Empty by default.
  quantile_dtype : str
      One of :data:`QUANTILE_DTYPES`: the dtype each margin's test
      quantiles are rounded to and stored in.
  """

  data_dir: Path
  stations: tuple[int, ...] | None
  selection: CaseSelection
  models: tuple[str, ...]
  responses: Mapping[str, ResponseSpec]
  estimators: tuple[EstimatorSpec, ...]
  copula_arms: tuple[str, ...]
  copula_margins: tuple[str, ...] | None = None
  copula_arms_by_margin: Mapping[str, tuple[str, ...]] = field(
    default_factory=dict
  )
  verify_data: bool = True
  modeled_responses: tuple[str, ...] | None = None
  derived_responses: tuple[str, ...] = ()
  magnus: str = METEOSWISS.name
  devices: Mapping[str, str] = field(default_factory=dict)
  n_levels: int = 199
  n_draws: int = 200
  sample_batch_size: int = 5000
  mixture_k: int = 5
  base_seed: int = 317
  batch_size: int | None = None
  train_pits: str = "in_sample"
  save_draws: bool = False
  copula_designs: Mapping[str, CopulaDesign] = field(default_factory=dict)
  quantile_dtype: str = "float64"

  def __post_init__(self) -> None:
    """Refuse a configuration no run could honor."""
    object.__setattr__(
      self, "responses", MappingProxyType(dict(self.responses))
    )
    object.__setattr__(
      self, "copula_designs", MappingProxyType(dict(self.copula_designs))
    )
    object.__setattr__(
      self,
      "copula_arms_by_margin",
      MappingProxyType(
        {m: tuple(a) for m, a in self.copula_arms_by_margin.items()}
      ),
    )
    object.__setattr__(self, "devices", MappingProxyType(dict(self.devices)))
    if self.stations is not None:
      if not self.stations or any(s < 0 for s in self.stations):
        raise ValueError(
          f"stations must be 'all' or a non-empty list of station ids >= 0; "
          f"got {list(self.stations)}"
        )
      _unique("stations", self.stations)
    if not self.responses:
      raise ValueError("responses must name at least one response; got none")
    self._check_estimators()
    if not self.models:
      raise ValueError("models must name at least one marginal model; got none")
    _unique("models", self.models)
    for name in self.models:
      self.marginal_model(name)
    if not all(self.copula_arms):
      raise ValueError(
        f"copula_arms must be non-empty names; got {list(self.copula_arms)}"
      )
    _unique("copula_arms", self.copula_arms)
    if self.copula_margins is not None:
      _unique("copula_margins", self.copula_margins)
      unknown = [m for m in self.copula_margins if m not in self.models]
      if unknown:
        raise ValueError(
          f"copula_margins must be drawn from the marginal models "
          f"{list(self.models)}; got {unknown}"
        )
    self._check_arms_by_margin()
    self._check_devices()
    self._check_modeled()
    self._check_designs()
    if self.quantile_dtype not in QUANTILE_DTYPES:
      raise ValueError(
        f"quantile_dtype must be one of {list(QUANTILE_DTYPES)}; got "
        f"{self.quantile_dtype!r}"
      )
    for name, low in (
      ("n_levels", 2),
      ("n_draws", 2),
      ("sample_batch_size", 1),
      ("mixture_k", 1),
    ):
      value = getattr(self, name)
      if value < low:
        raise ValueError(f"{name} must be at least {low}; got {value}")
    if self.batch_size is not None and self.batch_size < 1:
      raise ValueError(
        f"batch_size must be positive or omitted; got {self.batch_size}"
      )
    if self.train_pits not in TRAIN_PITS:
      raise ValueError(
        f"train_pits must be one of {list(TRAIN_PITS)}; got {self.train_pits!r}"
      )

  def _check_estimators(self) -> None:
    """Refuse an estimator a margin cannot be built from."""
    labels = [e.label for e in self.estimators]
    _unique("estimator labels", labels)
    external = [e.label for e in self.estimators if e.kind != "backend"]
    if external:
      raise EstimatorConfigError(
        f"estimators must be kind='backend', since every margin and copula "
        f"pair is an npcc backend; got the external {external}"
      )
    backends = [e.backend for e in self.estimators]
    if len(set(backends)) != len(backends):
      raise EstimatorConfigError(
        f"estimators must name each backend once, since a marginal model "
        f"names its estimator by backend; got {backends}"
      )

  def _check_arms_by_margin(self) -> None:
    """Refuse a per-margin arm list that no copula stage could run."""
    unknown = sorted(set(self.copula_arms_by_margin) - set(self.copula_models))
    if unknown:
      raise ValueError(
        f"copulas.arms_by_margin must name the copula stage's marginal models "
        f"{list(self.copula_models)}; got {unknown}"
      )
    for model, arms in self.copula_arms_by_margin.items():
      if not arms:
        raise ValueError(
          f"copulas.arms_by_margin.{model} must name at least one arm; to "
          f"run no arm on {model}, leave it out of copulas.margins"
        )
      _unique(f"copulas.arms_by_margin.{model}", arms)
      stray = [a for a in arms if a not in self.copula_arms]
      if stray:
        raise ValueError(
          f"copulas.arms_by_margin.{model} must be drawn from copulas.arms "
          f"{list(self.copula_arms)}; got {stray}"
        )

  def _check_devices(self) -> None:
    """Refuse a device for a backend no estimator uses, or a bad device."""
    unknown = sorted(set(self.devices) - set(self.estimators_by_backend))
    if unknown:
      raise ValueError(
        f"devices must name estimator backends "
        f"{sorted(self.estimators_by_backend)}; got {unknown}"
      )
    for backend, device in self.devices.items():
      try:
        torch.device(device)
      except RuntimeError as exc:
        raise ValueError(
          f"devices.{backend} must be a torch device; got {device!r}"
        ) from exc

  def _check_modeled(self) -> None:
    """Refuse a modeled subset or a derivation the data cannot support."""
    names = self.variables
    if self.modeled_responses is not None:
      _unique("modeled", self.modeled_responses)
      unknown = sorted(set(self.modeled_responses) - set(names))
      if not self.modeled_responses or unknown:
        raise ValueError(
          f"modeled must be a non-empty subset of the responses {list(names)}; "
          f"got {list(self.modeled_responses)}"
        )
    _unique("derived", self.derived_responses)
    for name in self.derived_responses:
      if name not in DERIVATIONS or name not in names:
        raise ValueError(
          f"derived must be drawn from {sorted(set(DERIVATIONS) & set(names))}, "
          f"the responses with a formula; got {name!r}"
        )
      if name in self.modeled:
        raise ValueError(f"{name} must be either modeled or derived, not both")
      missing = [v for v in DERIVATIONS[name] if v not in self.modeled]
      if missing:
        raise ValueError(
          f"{name} must be derived from modeled responses; its inputs "
          f"{missing} are not modeled"
        )
    Derivation(self.magnus)
    left = [v for v in names if v not in self.modeled and v not in self.derived]
    if left:
      raise ValueError(
        f"every response must be modeled or derived; got neither for {left}"
      )

  def _check_designs(self) -> None:
    """Refuse a design, or an arm's design, the data cannot support."""
    names = self.variables
    for name, design in self.copula_designs.items():
      where = f"copulas.designs.{name}"
      if not _DESIGN_NAME.fullmatch(name):
        raise ValueError(
          f"{where} must be named with lowercase letters, digits and "
          f"underscores; got {name!r}"
        )
      _unique(f"{where}.modeled", design.modeled)
      _unique(f"{where}.derived", design.derived)
      unknown = sorted(set(design.modeled) - set(names))
      if not design.modeled or unknown:
        raise ValueError(
          f"{where}.modeled must be a non-empty subset of the responses "
          f"{list(names)}; got {list(design.modeled)}"
        )
      for derived in design.derived:
        inputs = DERIVATIONS.get(derived)
        if inputs is None or derived not in names:
          raise ValueError(
            f"{where}.derived must be drawn from "
            f"{sorted(set(DERIVATIONS) & set(names))}, the responses with a "
            f"formula; got {derived!r}"
          )
        if derived in design.modeled:
          raise ValueError(
            f"{where}: {derived} must be either modeled or derived, not both"
          )
        missing = [v for v in inputs if v not in design.modeled]
        if missing:
          raise ValueError(
            f"{where}: {derived} must be derived from modeled responses; its "
            f"inputs {missing} are not modeled"
          )
      left = [
        v for v in names if v not in design.modeled and v not in design.derived
      ]
      if left:
        raise ValueError(
          f"{where} must model or derive every response, so its arms are "
          f"scored on all of them; got neither for {left}"
        )
    for arm in self.copula_arms:
      _, design = split_arm(arm)
      if design is not None and design not in self.copula_designs:
        raise ValueError(
          f"copulas.arms: {arm!r} must name a design of copulas.designs "
          f"{sorted(self.copula_designs)}; got {design!r}"
        )

  @property
  def margin_responses(self) -> tuple[str, ...]:
    """The responses the margins are fitted to, sorted as the data sorts them.

    Every response of the default design; otherwise every response the main
    design or another design of the copula stage models.
    """
    if self.is_default_design:
      return self.variables
    wanted = set(self.modeled)
    for design in self.copula_designs.values():
      wanted.update(design.modeled)
    return tuple(v for v in self.variables if v in wanted)

  def design(self, name: str | None) -> CopulaDesign:
    """The responses an arm of design ``name`` models and derives.

    Parameters
    ----------
    name : str or None
        A key of :attr:`copula_designs`; ``None`` for the main design,
        ``modeled`` and ``derived``.

    Returns
    -------
    CopulaDesign
        Its modeled and derived responses, sorted as the data sorts them.
    """
    if name is None:
      return CopulaDesign(modeled=self.modeled, derived=self.derived)
    if name not in self.copula_designs:
      raise ValueError(
        f"name must be a design of copulas.designs "
        f"{sorted(self.copula_designs)}; got {name!r}"
      )
    design = self.copula_designs[name]
    return CopulaDesign(
      modeled=tuple(v for v in self.variables if v in design.modeled),
      derived=tuple(v for v in self.variables if v in design.derived),
    )

  @property
  def variables(self) -> tuple[str, ...]:
    """Every response, sorted as the data sorts them."""
    return tuple(sorted(self.responses))

  @property
  def modeled(self) -> tuple[str, ...]:
    """The modeled responses, sorted as the data sorts them."""
    if self.modeled_responses is None:
      return self.variables
    return tuple(v for v in self.variables if v in self.modeled_responses)

  @property
  def derived(self) -> tuple[str, ...]:
    """The derived responses, sorted as the data sorts them."""
    return tuple(v for v in self.variables if v in self.derived_responses)

  @property
  def is_default_design(self) -> bool:
    """Whether every response is modeled, as in FLBuchner's design."""
    return self.modeled == self.variables and not self.derived

  @property
  def derivation(self) -> Derivation:
    """The Magnus formula of the derivations."""
    return Derivation(self.magnus)

  def _design(self) -> dict[str, object]:
    """The modeled subset's settings, recorded only off the default design.

    The default's signature and echo predate the option, so a run folder
    the default started still resumes.
    """
    if self.is_default_design:
      return {}
    return {
      "modeled": list(self.modeled),
      "derived": list(self.derived),
      "magnus": self.magnus,
    }

  def _outputs(self) -> dict[str, object]:
    """The settings added since, each recorded only off its default.

    The training PITs, the stored draws, the quantiles' dtype and the copula
    stage's other designs: the defaults' signature predates them all, so a
    run folder they started still resumes, and one that changes any does
    not.
    """
    out: dict[str, object] = {}
    if self.train_pits != "in_sample":
      out["train_pits"] = self.train_pits
    if self.save_draws:
      out["save_draws"] = True
    if self.quantile_dtype != "float64":
      out["quantile_dtype"] = self.quantile_dtype
    if self.copula_designs:
      out["copula_designs"] = {
        name: {
          "modeled": list(self.design(name).modeled),
          "derived": list(self.design(name).derived),
        }
        for name in sorted(self.copula_designs)
      }
    return out

  @property
  def copula_models(self) -> tuple[str, ...]:
    """The marginal models of the copula stage, in the configured order."""
    return self.models if self.copula_margins is None else self.copula_margins

  def copula_arms_for(self, model: str) -> tuple[str, ...]:
    """The arms the copula stage runs on one marginal model's sets.

    Parameters
    ----------
    model : str
        One of :attr:`copula_models`.

    Returns
    -------
    tuple of str
        Its entry in :attr:`copula_arms_by_margin`, or every arm when it has
        none; in the order of :attr:`copula_arms` either way.

    Raises
    ------
    ValueError
        If ``model`` is not a copula stage's marginal model.
    """
    if model not in self.copula_models:
      raise ValueError(
        f"model must be one of the copula stage's marginal models "
        f"{list(self.copula_models)}; got {model!r}"
      )
    chosen = self.copula_arms_by_margin.get(model, self.copula_arms)
    return tuple(a for a in self.copula_arms if a in chosen)

  @property
  def data_sha256(self) -> Mapping[str, str] | None:
    """The digest each input file must have; ``None`` when unchecked."""
    return DATA_SHA256 if self.verify_data else None

  @property
  def estimators_by_backend(self) -> Mapping[str, EstimatorSpec]:
    """Each estimator, by its backend's name."""
    return MappingProxyType({e.backend: e for e in self.estimators})

  @property
  def levels(self) -> np.ndarray:
    """The quantile levels, ``(k + 0.5) / n_levels`` for each ``k``."""
    return (np.arange(self.n_levels) + 0.5) / self.n_levels

  def marginal_model(self, name: str) -> MarginalModel:
    """Parse a marginal model's name.

    Parameters
    ----------
    name : str
        ``emos-seasonal`` or ``<backend>-<scope>``.

    Returns
    -------
    MarginalModel
        Its backend and scope.
    """
    if name == EMOS_MODEL:
      return MarginalModel(name=name, backend=None, scope="emos")
    backend, _, scope = name.rpartition("-")
    if backend not in self.estimators_by_backend or scope not in MARGIN_SCOPES:
      raise ValueError(
        f"models must be {EMOS_MODEL!r} or '<backend>-<scope>' with backend "
        f"in {sorted(self.estimators_by_backend)} and scope in "
        f"{list(MARGIN_SCOPES)}; got {name!r}"
      )
    return MarginalModel(name=name, backend=backend, scope=scope)

  def device_for(self, name: str) -> str | None:
    """The configured device of the backend ``name`` names.

    Parameters
    ----------
    name : str
        A backend, or a model or copula arm whose name contains one.

    Returns
    -------
    str or None
        The device of the longest configured backend name in ``name``;
        ``None`` if there is none, for the backend's own default.
    """
    matches = [b for b in self.devices if b in name]
    return self.devices[max(matches, key=len)] if matches else None

  def signature(self) -> str:
    """A hash of every setting that changes a stored result.

    Which stations, models and arms run, the devices and the inference batch
    size are left out: the first three choose what to compute, and the last
    two change results only at the rounding level.

    Returns
    -------
    str
        A SHA-256 hex digest.
    """
    payload = {
      "selection": _selection_echo(self.selection),
      "responses": {k: asdict(v) for k, v in sorted(self.responses.items())},
      "n_levels": self.n_levels,
      "estimators": {
        e.backend: e.estimator_id
        for e in sorted(self.estimators, key=lambda e: e.backend)
      },
      "n_draws": self.n_draws,
      "sample_batch_size": self.sample_batch_size,
      "mixture_k": self.mixture_k,
      "base_seed": self.base_seed,
      **self._design(),
      **self._outputs(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

  def echo(self) -> dict[str, object]:
    """The resolved configuration, as JSON-able values.

    Returns
    -------
    dict
        Every field, each estimator with its ``estimator_id``, and the
        signature.
    """
    return {
      "data_dir": str(self.data_dir),
      "stations": None if self.stations is None else list(self.stations),
      "selection": _selection_echo(self.selection),
      "models": list(self.models),
      "responses": {k: asdict(v) for k, v in self.responses.items()},
      "estimators": [
        {"label": e.label, "estimator_id": e.estimator_id, **e.canonical_dict()}
        for e in self.estimators
      ],
      "copula_arms": list(self.copula_arms),
      "copula_margins": list(self.copula_models),
      "copula_arms_by_margin": {
        m: list(self.copula_arms_for(m)) for m in self.copula_models
      },
      "verify_data": self.verify_data,
      "data_sha256": dict(DATA_SHA256) if self.verify_data else None,
      "devices": dict(self.devices),
      "n_levels": self.n_levels,
      "n_draws": self.n_draws,
      "sample_batch_size": self.sample_batch_size,
      "mixture_k": self.mixture_k,
      "base_seed": self.base_seed,
      "batch_size": self.batch_size,
      "train_pits": self.train_pits,
      "save_draws": self.save_draws,
      "quantile_dtype": self.quantile_dtype,
      "copula_designs": {
        name: {
          "modeled": list(self.design(name).modeled),
          "derived": list(self.design(name).derived),
        }
        for name in sorted(self.copula_designs)
      },
      "margin_responses": list(self.margin_responses),
      **self._design(),
      "signature": self.signature(),
    }


def _unique(name: str, values: Sequence[object]) -> None:
  """Refuse a list with a repeated entry."""
  if len(set(values)) != len(values):
    repeated = sorted({str(v) for v in values if values.count(v) > 1})
    raise ValueError(f"{name} must be unique; got {repeated} more than once")


def _selection_echo(selection: CaseSelection) -> dict[str, object]:
  """A selection as JSON-able values."""
  return {
    "train": [d.isoformat() for d in selection.train],
    "test": [d.isoformat() for d in selection.test],
    "lead_hours": selection.lead_hours,
    "init_hour": selection.init_hour,
    "min_cases": selection.min_cases,
  }


_TABLES: dict[str, tuple[str, ...]] = {
  "data": (
    "dir",
    "stations",
    "train",
    "test",
    "lead_hours",
    "init_hour",
    "min_cases",
    "verify",
  ),
  "margins": (
    "models",
    "n_levels",
    "batch_size",
    "train_pits",
    "quantile_dtype",
  ),
  "copulas": (
    "arms",
    "margins",
    "arms_by_margin",
    "n_draws",
    "sample_batch_size",
    "mixture_k",
    "save_draws",
    "designs",
  ),
}
#: The keys of ``[responses]`` that are settings rather than a response.
_DESIGN_KEYS = ("modeled", "derived", "magnus")
_TOP_LEVEL = (*_TABLES, "responses", "devices", "estimators", "base_seed")


def _table(raw: Mapping[str, object], name: str) -> dict[str, object]:
  """The table ``[name]``, refusing a key it does not define."""
  table = raw.get(name, {})
  if not isinstance(table, dict):
    raise TypeError(f"[{name}] must be a table; got {table!r}")
  allowed = _TABLES.get(name)
  if allowed is not None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
      raise ValueError(
        f"[{name}] must hold only {list(allowed)}; got unknown {unknown}"
      )
  return cast("dict[str, object]", table)


def _required(table: Mapping[str, object], name: str, key: str) -> object:
  if key not in table:
    raise ValueError(f"[{name}] must define {key!r}")
  return table[key]


def _date(value: object, name: str) -> dt.date:
  """A TOML date, or an ISO date string."""
  if isinstance(value, dt.datetime):
    raise TypeError(f"{name} must be a date without a time; got {value}")
  if isinstance(value, dt.date):
    return value
  if isinstance(value, str):
    try:
      return dt.date.fromisoformat(value)
    except ValueError as exc:
      raise ValueError(f"{name} must be an ISO date; got {value!r}") from exc
  raise ValueError(f"{name} must be a date; got {value!r}")


def _dates(value: object, name: str) -> tuple[dt.date, dt.date]:
  if not isinstance(value, list) or len(value) != 2:
    raise ValueError(f"{name} must be [first, last], two dates; got {value!r}")
  return _date(value[0], name), _date(value[1], name)


def _int(value: object, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer; got {value!r}")
  return value


def _strings(value: object, name: str) -> tuple[str, ...]:
  if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
    raise ValueError(f"{name} must be a list of strings; got {value!r}")
  return tuple(cast("list[str]", value))


def _stations(value: object) -> tuple[int, ...] | None:
  if value == "all":
    return None
  if not isinstance(value, list):
    raise TypeError(
      f"data.stations must be 'all' or a list of station ids; got {value!r}"
    )
  return tuple(_int(v, "data.stations") for v in value)


def _response(name: str, value: object) -> ResponseSpec:
  if not isinstance(value, dict):
    raise TypeError(
      f"responses.{name} must be a table such as {{ transform = 'identity' }}; "
      f"got {value!r}"
    )
  unknown = sorted(set(value) - {"transform", "scale", "boundary_nudge"})
  if unknown:
    raise ValueError(
      f"responses.{name} must hold only transform, scale and "
      f"boundary_nudge; got unknown {unknown}"
    )
  transform = _required(value, f"responses.{name}", "transform")
  if transform not in get_args(Transform):
    raise ValueError(
      f"responses.{name}.transform must be one of {list(_SUPPORTS)}; got "
      f"{transform!r}"
    )
  return ResponseSpec(
    transform=cast("Transform", transform),
    scale=float(cast("float", value.get("scale", 1.0))),
    boundary_nudge=float(cast("float", value.get("boundary_nudge", 0.0))),
  )


def _designs(value: object) -> dict[str, CopulaDesign]:
  """``[copulas.designs]``: each design's modeled and derived responses."""
  if not isinstance(value, dict):
    raise TypeError(
      f"copulas.designs must be a table of designs, such as "
      f"[copulas.designs.5d]; got {value!r}"
    )
  designs = {}
  for name, table in value.items():
    if not isinstance(table, dict):
      raise TypeError(
        f"copulas.designs.{name} must be a table with modeled and derived; "
        f"got {table!r}"
      )
    unknown = sorted(set(table) - {"modeled", "derived"})
    if unknown:
      raise ValueError(
        f"copulas.designs.{name} must hold only modeled and derived; got "
        f"unknown {unknown}"
      )
    designs[str(name)] = CopulaDesign(
      modeled=_strings(
        _required(table, f"copulas.designs.{name}", "modeled"),
        f"copulas.designs.{name}.modeled",
      ),
      derived=_strings(
        table.get("derived", []), f"copulas.designs.{name}.derived"
      ),
    )
  return designs


def load_config(path: str | Path) -> ApplicationConfig:
  """Read an :class:`ApplicationConfig` from a TOML file.

  Parameters
  ----------
  path : str or pathlib.Path
      The file.

  Returns
  -------
  ApplicationConfig
      The validated configuration.
  """
  with Path(path).open("rb") as fh:
    raw = tomllib.load(fh)
  unknown = sorted(set(raw) - set(_TOP_LEVEL))
  if unknown:
    raise ValueError(
      f"the configuration must hold only {list(_TOP_LEVEL)}; got unknown "
      f"{unknown}"
    )
  data = _table(raw, "data")
  margins = _table(raw, "margins")
  copulas = _table(raw, "copulas")
  responses = _table(raw, "responses")
  devices = _table(raw, "devices")
  raw_estimators = raw.get("estimators", [])
  if not isinstance(raw_estimators, list):
    raise EstimatorConfigError(
      "estimators must be an [[estimators]] array of tables"
    )
  selection = CaseSelection(
    train=_dates(_required(data, "data", "train"), "data.train"),
    test=_dates(_required(data, "data", "test"), "data.test"),
    **{
      key: _int(data[key], f"data.{key}")
      for key in ("lead_hours", "init_hour", "min_cases")
      if key in data
    },
  )
  # A key the file leaves out takes the dataclass default, written once.
  options: dict[str, Any] = {
    key: _int(table[key], f"{name}.{key}")
    for name, table, keys in (
      ("margins", margins, ("n_levels", "batch_size")),
      ("copulas", copulas, ("n_draws", "sample_batch_size", "mixture_k")),
    )
    for key in keys
    if key in table
  }
  if "base_seed" in raw:
    options["base_seed"] = _int(raw["base_seed"], "base_seed")
  if "verify" in data:
    if not isinstance(data["verify"], bool):
      raise TypeError(
        f"data.verify must be true or false; got {data['verify']!r}"
      )
    options["verify_data"] = data["verify"]
  if "train_pits" in margins:
    if not isinstance(margins["train_pits"], str):
      raise TypeError(
        f"margins.train_pits must be one of {list(TRAIN_PITS)}; got "
        f"{margins['train_pits']!r}"
      )
    options["train_pits"] = margins["train_pits"]
  if "quantile_dtype" in margins:
    if not isinstance(margins["quantile_dtype"], str):
      raise TypeError(
        f"margins.quantile_dtype must be one of {list(QUANTILE_DTYPES)}; got "
        f"{margins['quantile_dtype']!r}"
      )
    options["quantile_dtype"] = margins["quantile_dtype"]
  if "designs" in copulas:
    options["copula_designs"] = _designs(copulas["designs"])
  if "save_draws" in copulas:
    if not isinstance(copulas["save_draws"], bool):
      raise TypeError(
        f"copulas.save_draws must be true or false; got "
        f"{copulas['save_draws']!r}"
      )
    options["save_draws"] = copulas["save_draws"]
  if "margins" in copulas:
    options["copula_margins"] = _strings(copulas["margins"], "copulas.margins")
  if "arms_by_margin" in copulas:
    by_margin = copulas["arms_by_margin"]
    if not isinstance(by_margin, dict):
      raise TypeError(
        f"copulas.arms_by_margin must be a table of arm lists by marginal "
        f"model; got {by_margin!r}"
      )
    options["copula_arms_by_margin"] = {
      str(m): _strings(a, f"copulas.arms_by_margin.{m}")
      for m, a in by_margin.items()
    }
  tables = {k: v for k, v in responses.items() if k not in _DESIGN_KEYS}
  if "modeled" in responses:
    options["modeled_responses"] = _strings(
      responses["modeled"], "responses.modeled"
    )
  if "derived" in responses:
    options["derived_responses"] = _strings(
      responses["derived"], "responses.derived"
    )
  if "magnus" in responses:
    options["magnus"] = str(responses["magnus"])
  for key, value in devices.items():
    if not isinstance(value, str):
      raise TypeError(f"devices.{key} must be a string; got {value!r}")
  return ApplicationConfig(
    data_dir=Path(str(data.get("dir", DEFAULT_DATA_DIR))).expanduser(),
    stations=_stations(data.get("stations", "all")),
    selection=selection,
    models=_strings(_required(margins, "margins", "models"), "margins.models"),
    responses={k: _response(k, v) for k, v in tables.items()},
    estimators=tuple(
      _parse_estimator(cast("Mapping[str, object]", e)) for e in raw_estimators
    ),
    copula_arms=_strings(copulas.get("arms", []), "copulas.arms"),
    devices=cast("dict[str, str]", devices),
    **options,
  )
