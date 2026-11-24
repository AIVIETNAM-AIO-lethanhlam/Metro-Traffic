"""Naive one-hour-ahead references, read straight from the shared model frame (no fitting).

The three baselines of the group protocol (docs/problem_definition.md §3.5):
* Persistence      y_hat(t) = y(t-1)            (lag_1)
* Seasonal naive   y_hat(t) = y(t-24)           (lag_24, also the MASE scale)
* Hour-of-week     y_hat(t) = train mean of the same hour of the week (how_mean)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = {
    "Persistence (t-1)": "lag_1",
    "Seasonal naive (t-24)": "lag_24",
    "Hour-of-week mean": "how_mean",
}


def baseline_predictions(frame: pd.DataFrame, split: str) -> dict[str, np.ndarray]:
    part = frame[frame["split"] == split]
    fallback = part["how_mean"].to_numpy(float)
    out = {}
    for name, col in BASELINES.items():
        v = part[col].to_numpy(float)
        out[name] = np.where(np.isnan(v), fallback, v)
    return out
