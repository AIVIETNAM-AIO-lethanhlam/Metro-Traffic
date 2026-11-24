"""Point-forecast metrics (same definitions as the group's evaluate.py) and the Diebold-Mariano test."""
from __future__ import annotations

import math

import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def smape(y, p) -> float:
    """100 * mean(2|p - y| / (|y| + |p|)); a term is 0 when both are 0."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(den > 0, 2 * np.abs(p - y) / den, 0.0)
    return float(100 * t.mean())


def mase_scale(y_train, lag24_train) -> float:
    """In-sample MAE of the seasonal-naive (lag-24) forecast on the training targets."""
    y, l = np.asarray(y_train, float), np.asarray(lag24_train, float)
    ok = ~np.isnan(l)
    return float(np.mean(np.abs(y[ok] - l[ok])))


def all_metrics(y, p, scale: float | None = None) -> dict:
    out = dict(MAE=mae(y, p), RMSE=rmse(y, p), R2=r2(y, p), sMAPE=smape(y, p))
    if scale:
        out["MASE"] = out["MAE"] / scale
    return out


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def diebold_mariano(y, p_a, p_b, lags: int = 24) -> dict:
    """DM test on absolute errors, d_t = |e_a| - |e_b|, Newey-West variance with `lags` lags.

    mean_d < 0 means model A is more accurate. Two-sided p-value from the normal distribution.
    """
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(p_a, float)) - np.abs(y - np.asarray(p_b, float))
    n = len(d)
    dc = d - d.mean()
    var = np.dot(dc, dc) / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return dict(mean_d=float(d.mean()), dm_stat=float("nan"), p_value=float("nan"))
    stat = d.mean() / math.sqrt(var / n)
    return dict(mean_d=float(d.mean()), dm_stat=float(stat), p_value=float(2 * _norm_sf(abs(stat))))
