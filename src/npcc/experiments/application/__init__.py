"""The weather application: station-wise margins and conditional copulas.

Forecasts of five surface variables at 24 h lead, one station at a time:
marginal models fitted on 2017-2020 and scored on 2021, then copulas fitted
to the margins' probability integral transforms and scored jointly.
"""
