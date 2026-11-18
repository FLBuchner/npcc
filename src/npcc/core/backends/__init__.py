"""Concrete distributional-regression backends for npcc.

Each backend subclasses :class:`npcc.core.margin.ConditionalMargin` and
implements its two hooks: fitting, and predicting a table of conditional
quantiles.

Backends are constructed through the registry
(:mod:`npcc.core.registry`), never imported eagerly here, so that the
optional-extra backends (NGBoost, TabICL, LightGBM) only import their
third-party dependency when actually requested.
"""
