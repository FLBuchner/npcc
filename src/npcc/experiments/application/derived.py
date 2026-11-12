"""Responses derived from temperature, dew point and pressure.

In the MeteoSwiss data, relative humidity and the water vapor mixing ratio
are functions of the other three responses (``T`` and ``Td`` in degrees
Celsius, ``p`` in hPa):

- ``RH = 100 e_s(Td) / e_s(T)``, with a Magnus formula ``e_s``;
- ``w = 1000 * 0.622 e / (p - e)`` g/kg, with ``e = e_s(Td)``.

Which Magnus constants the data was computed with is not settled, so they
are one object, :class:`Magnus`, chosen by name (:data:`MAGNUS`):
Sonntag's (1990), one set for every temperature, which matches the data to
0.17 RH points at all 131 stations; and MeteoSwiss's piecewise set, one
above and one below 0 degrees Celsius, which Zanetta et al. (2023,
arXiv:2212.04487) use.

The data clips the dew point at the temperature: ``RH = 100`` exactly where
``Td = T``, and ``Td > T`` never occurs. Whether a draw with ``Td > T`` is
clipped the same way, ``min(Td, T)`` in both formulas, is a switch
(``clip``), off unless asked for; such draws are counted either way.

Five responses with two of them functions of the other three have no joint
density, so the application can model the triple and derive the other two
from its draws (``[responses] modeled`` and ``derived``). A derivation reads
the physical quantities ``T``, ``Td`` and ``p`` from the modeled responses
through :data:`QUANTITIES`, so a triple that models another function of
them names it there and nowhere else.
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

#: The physical quantity each modeled response is: ``T``, ``Td`` or ``p``.
QUANTITIES: Mapping[str, str] = MappingProxyType(
  {TEMPERATURE: "T", DEW_POINT: "Td", PRESSURE: "p"}
)
#: Each derivable response and the modeled responses it is computed from.
#: Both read the temperature: relative humidity divides by its ``e_s``, and
#: either clips the dew point at it when asked to.
DERIVATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
  {
    RELATIVE_HUMIDITY: (TEMPERATURE, DEW_POINT),
    MIXING_RATIO: (TEMPERATURE, DEW_POINT, PRESSURE),
  }
)


@dataclass(frozen=True)
class Derivation:
  """How derived responses are computed from the modeled ones.

  Attributes
  ----------
  magnus : str
      A key of :data:`MAGNUS`.
  clip_dew_point : bool
      Whether a dew point above the temperature is clipped at it, as the
      data clips, before RH and w are computed.
  """

  magnus: str = SONNTAG_1990.name
  clip_dew_point: bool = False

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
  t: np.ndarray, magnus: Magnus = SONNTAG_1990
) -> np.ndarray:
  """The saturation vapor pressure over water, in hPa.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  magnus : Magnus, optional
      The formula; Sonntag's by default.

  Returns
  -------
  numpy.ndarray
      ``e_s(t)``.
  """
  return magnus.saturation_vapor_pressure(t)


def _dew_point(t: np.ndarray, td: np.ndarray, clip: bool) -> np.ndarray:
  td = np.asarray(td, dtype=np.float64)
  return np.minimum(td, np.asarray(t, dtype=np.float64)) if clip else td


def relative_humidity(
  t: np.ndarray,
  td: np.ndarray,
  *,
  magnus: Magnus = SONNTAG_1990,
  clip: bool = False,
) -> np.ndarray:
  """Relative humidity in percent, ``100 e_s(Td) / e_s(T)``.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  td : numpy.ndarray
      Dew point in degrees Celsius, broadcast against ``t``.
  magnus : Magnus, optional
      The formula; Sonntag's by default.
  clip : bool, default=False
      Whether to clip the dew point at the temperature first, so RH <= 100.

  Returns
  -------
  numpy.ndarray
      The relative humidity.
  """
  td = _dew_point(t, td, clip)
  # The ratio first, so that Td = T gives exactly 100.
  e = magnus.saturation_vapor_pressure(td)
  return 100.0 * (e / magnus.saturation_vapor_pressure(t))


def mixing_ratio(
  t: np.ndarray,
  td: np.ndarray,
  p: np.ndarray,
  *,
  magnus: Magnus = SONNTAG_1990,
  clip: bool = False,
) -> np.ndarray:
  """The water vapor mixing ratio in g/kg, ``1000 * 0.622 e / (p - e)``.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius; read only to clip the dew point.
  td : numpy.ndarray
      Dew point in degrees Celsius; ``e = e_s(td)``.
  p : numpy.ndarray
      Surface pressure in hPa.
  magnus : Magnus, optional
      The formula; Sonntag's by default.
  clip : bool, default=False
      Whether to clip the dew point at the temperature first.

  Returns
  -------
  numpy.ndarray
      The mixing ratio.
  """
  e = magnus.saturation_vapor_pressure(_dew_point(t, td, clip))
  return MIXING_FACTOR * e / (np.asarray(p, dtype=np.float64) - e)


def saturation_dew_point(
  t: np.ndarray, nudge: float, magnus: Magnus = SONNTAG_1990
) -> np.ndarray:
  """The dew point at which RH is ``100 - nudge`` at temperature ``t``.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  nudge : float
      RH points below saturation, in ``[0, 100)``.
  magnus : Magnus, optional
      The formula; Sonntag's by default.

  Returns
  -------
  numpy.ndarray
      ``e_s^-1((1 - nudge / 100) e_s(t))``, just below ``t``.
  """
  e = (1.0 - nudge / 100.0) * magnus.saturation_vapor_pressure(t)
  return magnus.dew_point(e)


def nudge_saturated(
  t: np.ndarray,
  td: np.ndarray,
  nudge: float,
  magnus: Magnus = SONNTAG_1990,
) -> tuple[np.ndarray, int]:
  """Count the dew points equal to their temperature, and move them if asked.

  Parameters
  ----------
  t : numpy.ndarray, shape (n,)
      Temperature in degrees Celsius.
  td : numpy.ndarray, shape (n,)
      Dew point in degrees Celsius.
  nudge : float
      RH points below saturation to move them to; 0 moves none.
  magnus : Magnus, optional
      The formula; Sonntag's by default.

  Returns
  -------
  td : numpy.ndarray, shape (n,)
      A copy, with ``td == t`` replaced by :func:`saturation_dew_point`
      when ``nudge`` is positive.
  count : int
      How many have ``td == t``, whatever ``nudge`` is.
  """
  out = np.array(td, dtype=np.float64, copy=True)
  t = np.asarray(t, dtype=np.float64)
  saturated = out == t
  if nudge > 0.0:
    out[saturated] = saturation_dew_point(t[saturated], nudge, magnus)
  return out, int(saturated.sum())


def derive(
  values: np.ndarray,
  variables: Sequence[str],
  derived: Sequence[str],
  derivation: Derivation | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """The derived responses of every row of ``values``, and where Td > T.

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
      The Magnus formula and the clip; Sonntag's, unclipped, by default.

  Returns
  -------
  derived_values : numpy.ndarray, shape (..., len(derived))
      The derived responses.
  above : numpy.ndarray of bool, shape (...)
      Where the dew point exceeds the temperature, whether or not it was
      clipped.
  """
  derivation = derivation or Derivation()
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
  shape = values.shape[:-1]
  if not derived:
    return np.empty((*shape, 0)), np.zeros(shape, dtype=bool)
  q = {
    QUANTITIES[v]: values[..., names.index(v)] for v in names if v in QUANTITIES
  }
  formula, clip = derivation.formula, derivation.clip_dew_point
  out = []
  for name in derived:
    if name == RELATIVE_HUMIDITY:
      out.append(relative_humidity(q["T"], q["Td"], magnus=formula, clip=clip))
    else:
      out.append(
        mixing_ratio(q["T"], q["Td"], q["p"], magnus=formula, clip=clip)
      )
  return np.stack(out, axis=-1), q["Td"] > q["T"]
