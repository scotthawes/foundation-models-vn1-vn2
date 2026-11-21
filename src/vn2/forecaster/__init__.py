"""Forecasters for VN2.

Every forecaster has the competition's policy signature and returns a table of predicted demand:
one row per Store x Product, one column per week ahead, in the same row order as the input. They
differ only in how they get the number; the ordering policy downstream is shared, so a difference
in score is a difference in forecasting.

The foundation-model arms come in two forms. `tabpfn` and `chronos` run locally on the open weights;
`tabpfn_api` and `timegpt` call the hosted services instead. On this machine the local TabPFN arm is
the one that does not fit (3.73 GB MPS budget, 531.7 s per fit), so the back-test uses the two API
arms and keeps the local ones as a cross-check.

Every arm imports its heavy dependency lazily, so the cheap arms can be run without installing
either model library, and importing this package never needs an API key.
"""
from .naive import naive_mean, seasonal_naive
from .seasonal import seasonal_moving_average

__all__ = [
    "calibrated_chronos",
    "chronos",
    "chronos_quantiles",
    "naive_mean",
    "seasonal_moving_average",
    "seasonal_naive",
    "tabpfn",
    "tabpfn_api",
    "timegpt",
]


def tabpfn(**kwargs):
    """The TabPFN arm; see `vn2.forecaster.tabpfn_arm`."""
    from .tabpfn_arm import tabpfn as factory

    return factory(**kwargs)


def chronos(**kwargs):
    """The Chronos arm; see `vn2.forecaster.chronos_arm`."""
    from .chronos_arm import chronos as factory

    return factory(**kwargs)


def calibrated_chronos(**kwargs):
    """Chronos with a causal per-series bias correction; see `vn2.forecaster.chronos_arm`."""
    from .chronos_arm import calibrated_chronos as factory

    return factory(**kwargs)


def chronos_quantiles(**kwargs):
    """The raw nine-quantile Chronos forecast; see `vn2.forecaster.chronos_arm`."""
    from .chronos_arm import chronos_quantiles as factory

    return factory(**kwargs)


def tabpfn_api(**kwargs):
    """The TabPFN arm through the hosted API; see `vn2.forecaster.tabpfn_api_arm`."""
    from .tabpfn_api_arm import tabpfn_api as factory

    return factory(**kwargs)


def timegpt(**kwargs):
    """The TimeGPT arm; see `vn2.forecaster.timegpt_arm`."""
    from .timegpt_arm import timegpt as factory

    return factory(**kwargs)
