"""Responses derived from temperature, relative humidity and pressure.

In the MeteoSwiss data, temperature ``T`` (degrees Celsius), relative
humidity ``RH`` (percent) and surface pressure ``p`` (hPa) are measured,
stored exactly on a 0.1 grid; the dew point ``Td`` and the water vapor
mixing ratio ``w`` are computed from them:

- ``e = RH / 100 e_s(T)``, the vapor pressure, with a Magnus formula ``e_s``;
- ``Td = e_s^-1(e)``;
- ``w = 1000 * 0.622 e / (p - e)`` g/kg.

The Magnus formula is one object, :class:`Magnus`, chosen by name
(:data:`MAGNUS`): MeteoSwiss's piecewise formula, one set of constants at
and above 0 degrees Celsius and one below, as Zanetta et al. (2023,
arXiv:2212.04487) use, and Sonntag's (1990), one set everywhere. Since
``RH <= 100``, ``Td <= T`` follows.

Five responses with two of them functions of the other three have no joint
density, so the application can model the measured triple and derive the
other two from its draws (``[responses] modeled`` and ``derived``). A
derivation reads the physical quantities ``T``, ``RH`` and ``p`` from the
modeled responses through :data:`QUANTITIES`, so a triple that models
another function of them names it there and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

TEMPERATURE = "obs:air_temperature"
DEW_POINT = "obs:dew_point_temperature"
PRESSURE = "obs:surface_air_pressure"
RELATIVE_HUMIDITY = "obs:relative_humidity"
MIXING_RATIO = "obs:water_vapor_mixing_ratio"

#: 1000 times the ratio of the gas constants of dry air and water vapor:
#: the mixing ratio is ``MIXING_FACTOR e / (p - e)`` g/kg.
MIXING_FACTOR: float = 1000.0 * 0.622


@dataclass(frozen=True)
class Magnus:
  """A Magnus formula, ``e_s(t) = c exp(a t / (b + t))`` hPa, t in Celsius.

  Attributes
  ----------
  name : str
      How a configuration names it.
  a, b, c : float
      The constants at and above 0 degrees Celsius (everywhere, without
      ``below``).
  below : tuple of float or None
      ``(a, b, c)`` below 0 degrees Celsius, for a piecewise formula.
  """

  name: str
  a: float
  b: float
  c: float
  below: tuple[float, float, float] | None = None

  def _pieces(
    self,
  ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    above = (self.a, self.b, self.c)
    return above, self.below or above

  def saturation_vapor_pressure(self, t: np.ndarray) -> np.ndarray:
    """``e_s(t)`` in hPa.

    Parameters
    ----------
    t : numpy.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    numpy.ndarray
        The saturation vapor pressure over water.
    """
    t = np.asarray(t, dtype=np.float64)
    (a, b, c), (a2, b2, c2) = self._pieces()
    out = c * np.exp(a * t / (b + t))
    if self.below is not None:
      out = np.where(t >= 0.0, out, c2 * np.exp(a2 * t / (b2 + t)))
    return out

  def dew_point(self, e: np.ndarray) -> np.ndarray:
    """The temperature whose ``e_s`` is ``e``: the inverse of the formula.

    For a piecewise formula, the branch is the one whose result has its own
    sign. MeteoSwiss's branches overlap, ``e_s`` jumping from 6.108 just
    below 0 to 6.107 at 0, so an ``e`` in between has a dew point on each
    side, 0.0023 degrees Celsius apart at most; the one at or above 0 is
    taken.

    Parameters
    ----------
    e : numpy.ndarray
        Vapor pressure in hPa.

    Returns
    -------
    numpy.ndarray
        The dew point in degrees Celsius.
    """
    e = np.asarray(e, dtype=np.float64)
    (a, b, c), (a2, b2, c2) = self._pieces()
    g = np.log(e / c)
    t = b * g / (a - g)
    if self.below is not None:
      g2 = np.log(e / c2)
      t = np.where(t >= 0.0, t, b2 * g2 / (a2 - g2))
    return t


#: Sonntag (1990), Zeitschrift fuer Meteorologie 40, 340-344.
SONNTAG_1990 = Magnus("sonntag1990", a=17.62, b=243.12, c=6.112)
#: MeteoSwiss's piecewise formula, as Zanetta et al. (2023) use it.
METEOSWISS = Magnus(
  "meteoswiss", a=17.368, b=238.83, c=6.107, below=(17.856, 245.52, 6.108)
)
#: Every Magnus formula a configuration may name.
MAGNUS: Mapping[str, Magnus] = MappingProxyType(
  {m.name: m for m in (SONNTAG_1990, METEOSWISS)}
)

#: The physical quantity each modeled response is: ``T``, ``RH`` or ``p``.
QUANTITIES: Mapping[str, str] = MappingProxyType(
  {TEMPERATURE: "T", RELATIVE_HUMIDITY: "RH", PRESSURE: "p"}
)
#: Each derivable response and the modeled responses it is computed from.
DERIVATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
  {
    DEW_POINT: (TEMPERATURE, RELATIVE_HUMIDITY),
    MIXING_RATIO: (TEMPERATURE, RELATIVE_HUMIDITY, PRESSURE),
  }
)


@dataclass(frozen=True)
class Derivation:
  """How derived responses are computed from the modeled ones.

  Attributes
  ----------
  magnus : str
      A key of :data:`MAGNUS`.
  """

  magnus: str = "meteoswiss"

  def __post_init__(self) -> None:
    """Refuse a Magnus formula no one defined."""
    if self.magnus not in MAGNUS:
      raise ValueError(
        f"magnus must be one of {list(MAGNUS)}; got {self.magnus!r}"
      )

  @property
  def formula(self) -> Magnus:
    """The Magnus formula."""
    return MAGNUS[self.magnus]


def saturation_vapor_pressure(
  t: np.ndarray, magnus: Magnus = METEOSWISS
) -> np.ndarray:
  """The saturation vapor pressure over water, in hPa.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  magnus : Magnus, optional
      The formula; MeteoSwiss's by default.

  Returns
  -------
  numpy.ndarray
      ``e_s(t)``.
  """
  return magnus.saturation_vapor_pressure(t)


def vapor_pressure(
  t: np.ndarray, rh: np.ndarray, magnus: Magnus = METEOSWISS
) -> np.ndarray:
  """The vapor pressure ``e = RH / 100 e_s(T)``, in hPa.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  rh : numpy.ndarray
      Relative humidity in percent.
  magnus : Magnus, optional
      The formula; MeteoSwiss's by default.

  Returns
  -------
  numpy.ndarray
      ``e``.
  """
  return (
    np.asarray(rh, dtype=np.float64)
    / 100.0
    * (magnus.saturation_vapor_pressure(t))
  )


def dew_point(
  t: np.ndarray, rh: np.ndarray, magnus: Magnus = METEOSWISS
) -> np.ndarray:
  """The dew point ``Td = e_s^-1(RH / 100 e_s(T))``, in degrees Celsius.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  rh : numpy.ndarray
      Relative humidity in percent.
  magnus : Magnus, optional
      The formula; MeteoSwiss's by default.

  Returns
  -------
  numpy.ndarray
      ``Td``, at most ``T`` where ``RH <= 100``.
  """
  return magnus.dew_point(vapor_pressure(t, rh, magnus))


def mixing_ratio(
  t: np.ndarray,
  rh: np.ndarray,
  p: np.ndarray,
  magnus: Magnus = METEOSWISS,
) -> np.ndarray:
  """The water vapor mixing ratio ``1000 * 0.622 e / (p - e)``, in g/kg.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  rh : numpy.ndarray
      Relative humidity in percent.
  p : numpy.ndarray
      Surface pressure in hPa.
  magnus : Magnus, optional
      The formula; MeteoSwiss's by default.

  Returns
  -------
  numpy.ndarray
      ``w``.
  """
  e = vapor_pressure(t, rh, magnus)
  return MIXING_FACTOR * e / (np.asarray(p, dtype=np.float64) - e)


def derive(
  values: np.ndarray,
  variables: Sequence[str],
  derived: Sequence[str],
  derivation: Derivation | None = None,
) -> np.ndarray:
  """The derived responses of every row of ``values``.

  Parameters
  ----------
  values : numpy.ndarray, shape (..., d)
      The modeled responses, last axis in the order of ``variables``.
  variables : sequence of str
      The modeled responses' names.
  derived : sequence of str
      The responses to derive, each a key of :data:`DERIVATIONS` whose
      inputs are all in ``variables``.
  derivation : Derivation, optional
      The Magnus formula; MeteoSwiss's by default.

  Returns
  -------
  numpy.ndarray, shape (..., len(derived))
      The derived responses, in the order of ``derived``.
  """
  formula = (derivation or Derivation()).formula
  names = list(variables)
  for name in derived:
    if name not in DERIVATIONS:
      raise ValueError(
        f"derived must be drawn from {list(DERIVATIONS)}; got {name!r}"
      )
    missing = [v for v in DERIVATIONS[name] if v not in names]
    if missing:
      raise ValueError(
        f"{name} must be derived from modeled {list(DERIVATIONS[name])}; "
        f"{missing} are not modeled"
      )
  if not derived:
    return np.empty((*values.shape[:-1], 0))
  q = {
    QUANTITIES[v]: values[..., names.index(v)] for v in names if v in QUANTITIES
  }
  out = [
    dew_point(q["T"], q["RH"], formula)
    if name == DEW_POINT
    else mixing_ratio(q["T"], q["RH"], q["p"], formula)
    for name in derived
  ]
  return np.stack(out, axis=-1)


def residuals(
  values: np.ndarray,
  variables: Sequence[str],
  derived: Sequence[str],
  observed: np.ndarray,
  derivation: Derivation | None = None,
) -> dict[str, dict[str, float]]:
  """How far the derivation is from observed derived responses.

  Parameters
  ----------
  values : numpy.ndarray, shape (n, d)
      Observed modeled responses, in the order of ``variables``.
  variables : sequence of str
      The modeled responses' names.
  derived : sequence of str
      The derived responses' names.
  observed : numpy.ndarray, shape (n, len(derived))
      Their observations, in the order of ``derived``.
  derivation : Derivation, optional
      The Magnus formula; MeteoSwiss's by default.

  Returns
  -------
  dict of str to dict
      Per derived response: ``n``, the residual's (derived less observed)
      ``mean`` and ``sd``, and ``max_abs``, its largest magnitude.
  """
  residual = derive(values, variables, derived, derivation) - observed
  return {
    name: {
      "n": float(len(residual)),
      "mean": float(residual[:, j].mean()),
      "sd": float(residual[:, j].std()),
      "max_abs": float(np.abs(residual[:, j]).max()),
    }
    for j, name in enumerate(derived)
  }
