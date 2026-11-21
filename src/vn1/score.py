"""The organisers' score, verbatim: `(sum |error| + |sum error|) / sum actual`.

The first term is ordinary absolute error; the second is a bias penalty, so two forecasts with the
same absolute error are not equal if one is systematically high. A zero forecast scores 2.0, a
random one about 1.76, and the organisers' own MA12 dummy 0.6597 on the full 15,053 series
(`tests/test_vn1.py` pins that number).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score(submission: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Score a submission against actual sales. Lower is better; 0.0 is perfect."""
    if not submission.index.equals(actual.index):
        raise ValueError("submission and actual have different series")
    if not submission.columns.equals(actual.columns):
        raise ValueError("submission and actual have different weeks")

    error = submission.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    actual_total = actual.to_numpy(dtype=float).sum()
    if not np.isfinite(error).all():
        raise ValueError("submission contains NaN or infinite values")
    return float((np.abs(error).sum() + abs(error.sum())) / actual_total)
