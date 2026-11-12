"""Responses derived from temperature, dew point and pressure.

In the MeteoSwiss data, relative humidity and the water vapor mixing ratio
are functions of the other three responses:

- ``RH = 100 e_s(Td) / e_s(T)``, with the Magnus formula
  ``e_s(t) = 6.112 exp(17.62 t / (243.12 + t))`` hPa (t in degrees Celsius),
  Sonntag's constants (Sonntag, 1990, Zeitschrift fuer Meteorologie 40,
  340-344). It matches the data to 0.17 RH points at all 131 stations.
- ``w = 622 e / (p - e)`` g/kg, with ``e = e_s(Td)`` and ``p`` in hPa, 622
  being 1000 times the ratio of the gas constants of dry air and water vapor.
  It matches the data with a log-residual SD of 0.0017.

The data clips the dew point at the temperature, so ``RH = 100`` exactly
where ``Td = T``, and never has ``Td > T``. A draw with ``Td > T`` is
clipped the same way, ``min(Td, T)`` in both formulas, and counted.

Five responses with two of them functions of the other three have no joint
density, so the application can model the triple and derive the other two
from its draws (``[responses] modeled`` and ``derived``).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

#: The Magnus formula's constants, Sonntag (1990): ``e_s(t) = MAGNUS_E0
#: exp(MAGNUS_A t / (MAGNUS_B + t))`` hPa, ``t`` in degrees Celsius.
MAGNUS_E0: float = 6.112
MAGNUS_A: float = 17.62
MAGNUS_B: float = 243.12
#: 1000 R_d / R_v in g/kg: the mixing ratio is ``MIXING_FACTOR e / (p - e)``.
MIXING_FACTOR: float = 622.0

TEMPERATURE = "obs:air_temperature"
DEW_POINT = "obs:dew_point_temperature"
PRESSURE = "obs:surface_air_pressure"
RELATIVE_HUMIDITY = "obs:relative_humidity"
MIXING_RATIO = "obs:water_vapor_mixing_ratio"

#: Each derivable response and the modeled responses it is computed from.
#: The mixing ratio needs the temperature too, to clip the dew point at it.
DERIVATIONS = MappingProxyType(
  {
    RELATIVE_HUMIDITY: (TEMPERATURE, DEW_POINT),
    MIXING_RATIO: (TEMPERATURE, DEW_POINT, PRESSURE),
  }
)


def saturation_vapor_pressure(t: np.ndarray) -> np.ndarray:
  """The Magnus saturation vapor pressure over water, in hPa.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.

  Returns
  -------
  numpy.ndarray
      ``e_s(t)``.
  """
  t = np.asarray(t, dtype=np.float64)
  return MAGNUS_E0 * np.exp(MAGNUS_A * t / (MAGNUS_B + t))


def relative_humidity(t: np.ndarray, td: np.ndarray) -> np.ndarray:
  """Relative humidity in percent, the dew point clipped at the temperature.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  td : numpy.ndarray
      Dew point in degrees Celsius, broadcast against ``t``.

  Returns
  -------
  numpy.ndarray
      ``100 e_s(min(td, t)) / e_s(t)``, at most 100.
  """
  t = np.asarray(t, dtype=np.float64)
  td = np.minimum(np.asarray(td, dtype=np.float64), t)
  return 100.0 * np.exp(
    MAGNUS_A * td / (MAGNUS_B + td) - MAGNUS_A * t / (MAGNUS_B + t)
  )


def mixing_ratio(t: np.ndarray, td: np.ndarray, p: np.ndarray) -> np.ndarray:
  """The water vapor mixing ratio in g/kg, the dew point clipped at ``t``.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  td : numpy.ndarray
      Dew point in degrees Celsius.
  p : numpy.ndarray
      Surface pressure in hPa.

  Returns
  -------
  numpy.ndarray
      ``622 e / (p - e)`` with ``e = e_s(min(td, t))``.
  """
  td = np.minimum(np.asarray(td, dtype=np.float64), np.asarray(t))
  e = saturation_vapor_pressure(td)
  return MIXING_FACTOR * e / (np.asarray(p, dtype=np.float64) - e)


def saturation_dew_point(t: np.ndarray, nudge: float) -> np.ndarray:
  """The dew point at which the Magnus RH is ``100 - nudge`` at ``t``.

  Solves ``100 e_s(td) / e_s(t) = 100 - nudge`` for ``td``: the dew point a
  saturated case is moved to, as an RH of exactly 100 is moved to
  ``100 - nudge``.

  Parameters
  ----------
  t : numpy.ndarray
      Temperature in degrees Celsius.
  nudge : float
      RH points below saturation, in ``(0, 100)``.

  Returns
  -------
  numpy.ndarray
      A dew point just below ``t``.
  """
  t = np.asarray(t, dtype=np.float64)
  g = np.log1p(-nudge / 100.0) + MAGNUS_A * t / (MAGNUS_B + t)
  return MAGNUS_B * g / (MAGNUS_A - g)


def nudge_saturated(
  t: np.ndarray, td: np.ndarray, nudge: float
) -> tuple[np.ndarray, int]:
  """Move every dew point equal to its temperature just below saturation.

  Parameters
  ----------
  t : numpy.ndarray, shape (n,)
      Temperature in degrees Celsius.
  td : numpy.ndarray, shape (n,)
      Dew point in degrees Celsius.
  nudge : float
      RH points below saturation; 0 leaves every dew point as it is.

  Returns
  -------
  td : numpy.ndarray, shape (n,)
      A copy, with ``td == t`` replaced by :func:`saturation_dew_point`.
  count : int
      How many were replaced.
  """
  out = np.array(td, dtype=np.float64, copy=True)
  if nudge == 0.0:
    return out, 0
  saturated = out == np.asarray(t)
  out[saturated] = saturation_dew_point(np.asarray(t)[saturated], nudge)
  return out, int(saturated.sum())


def derive(
  values: np.ndarray, variables: Sequence[str], derived: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
  """The derived responses of every row of ``values``, and which clipped.

  Parameters
  ----------
  values : numpy.ndarray, shape (..., d)
      The modeled responses, last axis in the order of ``variables``.
  variables : sequence of str
      The modeled responses' names.
  derived : sequence of str
      The responses to derive, each a key of :data:`DERIVATIONS` whose
      inputs are all in ``variables``.

  Returns
  -------
  derived_values : numpy.ndarray, shape (..., len(derived))
      The derived responses.
  clipped : numpy.ndarray of bool, shape (...)
      Where the dew point exceeded the temperature and was clipped at it.
  """
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
  columns = {v: values[..., names.index(v)] for v in names}
  out = []
  for name in derived:
    t, td = columns[TEMPERATURE], columns[DEW_POINT]
    if name == RELATIVE_HUMIDITY:
      out.append(relative_humidity(t, td))
    else:
      out.append(mixing_ratio(t, td, columns[PRESSURE]))
  shape = values.shape[:-1]
  if not derived:
    return np.empty((*shape, 0)), np.zeros(shape, dtype=bool)
  clipped = columns[DEW_POINT] > columns[TEMPERATURE]
  return np.stack(out, axis=-1), clipped
