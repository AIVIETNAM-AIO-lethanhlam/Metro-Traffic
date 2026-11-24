"""Non deep-learning baselines: daily-naive, seasonal-naive (weekly) and Ridge regression."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .features import Dataset


def naive_forecast(ds: Dataset, ix, lag: int) -> np.ndarray:
    """y_hat[t+h] = y[t+h-lag]; lag=24 -> same hour yesterday, lag=168 -> same hour last week."""
    return ds.lagged(ix, lag)


def ridge_design(ds: Dataset, ix) -> np.ndarray:
    """168 scaled traffic lags + weather at the origin + calendar of the next H hours."""
    ix = np.asarray(ix)
    lags = ds.y_sc[ix[:, None] + ds.past_off]
    n_w = len(ds.weather_feats)
    now = ds.X_past[ix][:, 1:1 + n_w]
    fut = ds.X_fut[ix[:, None] + ds.fut_off].reshape(len(ix), -1)
    return np.hstack([lags, now, fut])


def fit_ridge(ds: Dataset, alphas=(0.1, 1, 10, 100), verbose: bool = True):
    """Select alpha on the validation set. Returns (model, alpha, val_MAE)."""
    Xtr, ytr = ridge_design(ds, ds.idx_train), ds.y_sc[ds.target_pos(ds.idx_train)]
    Xva, yva = ridge_design(ds, ds.idx_val), ds.targets(ds.idx_val)
    best = (None, None, np.inf)
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        mae = float(np.mean(np.abs(np.clip(m.predict(Xva) * ds.y_std + ds.y_mean, 0, None) - yva)))
        if verbose:
            print(f"  Ridge alpha={a:<6} val MAE={mae:.1f}")
        if mae < best[2]:
            best = (m, a, mae)
    return best


def ridge_predict(ds: Dataset, model, ix) -> np.ndarray:
    return model.predict(ridge_design(ds, ix)) * ds.y_std + ds.y_mean
