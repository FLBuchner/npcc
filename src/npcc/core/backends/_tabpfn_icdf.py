"""Invert one probability per TabPFN logit row."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from functools import lru_cache
from typing import Any, cast

import torch

# AST fingerprints exclude documentation and decorators, but include every
# operation. Unknown implementations keep their native scalar semantics.
# Ordinary: TabPFN 8.0.8 and upstream 89cc3a35; tails: upstream 89cc3a35.
ORDINARY = "678709bf3161700301a30be440ac63ab046a7b38812fdf25f5e625ea1070abbb"
TAILS = "04be7efc67a869c4574aba4ea2bae32f9e6412c11323b7709e40823a2e5acd4b"


@lru_cache(maxsize=32)
def inversion_kind(method: object) -> str | None:
  """Recognize reviewed implementations, failing closed if source is absent.

  Parameters
  ----------
  method : object
    Unbound native inversion function.

  Returns
  -------
  str or None
    Reviewed inversion kind, or ``None`` for scalar evaluation.
  """
  try:
    if not inspect.isfunction(method):
      return None
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
      return None
    function.decorator_list = []
    if isinstance(function.body[0], ast.Expr) and isinstance(
      function.body[0].value, ast.Constant
    ):
      function.body.pop(0)
    fingerprint = hashlib.sha256(ast.dump(function).encode()).hexdigest()
  except (OSError, TypeError, SyntaxError):
    return None
  return {ORDINARY: "ordinary", TAILS: "tails"}.get(fingerprint)


def rowwise_icdf(
  criterion: Any,  # noqa: ANN401 - native criteria expose version-specific members
  logits: torch.Tensor,
  p: torch.Tensor,
) -> torch.Tensor:
  """Return one native quantile per row without expanding probabilities.

  Parameters
  ----------
  criterion : object
    Native TabPFN predictive criterion.
  logits : torch.Tensor
    Logits with shape ``(N, K)`` on the criterion's device.
  p : torch.Tensor
    Row probabilities with shape ``(N,)``.

  Returns
  -------
  torch.Tensor
    Quantiles with shape ``(N,)``, placed and typed like ``p``.
  """
  method = criterion.icdf
  kind = inversion_kind(getattr(method, "__func__", method))
  if kind is None:
    return (
      torch.stack(
        [
          criterion.icdf(logits[i : i + 1], float(p[i].item()))
          .reshape(-1)[0]
          .to(device=p.device, dtype=p.dtype)
          for i in range(p.numel())
        ]
      )
      if p.numel()
      else p.clone()
    )

  probs = logits.softmax(-1)
  cumulative = probs.cumsum(-1)
  # Native ordinary inversion builds its search values with default-dtype
  # ones, even for float64 logits. Scalar subtraction instead uses the
  # cumulative tensor's dtype. Preserve both roundings independently.
  search_dtype = logits.dtype if kind == "tails" else torch.get_default_dtype()
  search_p = p.to(device=logits.device, dtype=search_dtype)
  before = torch.cat(
    (torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=-1
  )
  if kind == "ordinary":
    before = before.to(torch.promote_types(logits.dtype, search_dtype))
  probability = p.to(device=logits.device, dtype=before.dtype)
  index = torch.searchsorted(cumulative, search_p[:, None].contiguous())
  index = index.clamp(0, logits.shape[-1] - 1)
  rest = probability - before.gather(1, index).squeeze(1)
  mass = probs.gather(1, index).squeeze(1)
  index = index.squeeze(1)
  left = criterion.borders[index]
  if kind == "ordinary":
    values = left + (criterion.borders[index + 1] - left) * rest / mass
  else:
    conditional = (rest / mass).clamp(0.0, 1.0)
    values = left + criterion.bucket_widths[index] * conditional
    left_normal = criterion.halfnormal_with_p_weight_before(
      criterion.bucket_widths[0]
    )
    right_normal = criterion.halfnormal_with_p_weight_before(
      criterion.bucket_widths[-1]
    )
    values = torch.where(
      index == 0,
      criterion.borders[1] - left_normal.icdf(1.0 - conditional),
      values,
    )
    values = torch.where(
      index == logits.shape[-1] - 1,
      criterion.borders[-2] + right_normal.icdf(conditional),
      values,
    )
    # Native endpoint checks use the original Python scalar, before rounding
    # it to the logits dtype for arithmetic.
    endpoints = p.to(device=logits.device)
    values = torch.where(endpoints == 0, -torch.inf, values)
    values = torch.where(endpoints == 1, torch.inf, values)
  return cast("torch.Tensor", values.to(device=p.device, dtype=p.dtype))
