"""168-hour input windows for the one-hour-ahead task (NumPy only, no PyTorch).

A sample is a model-frame row at target hour t. Its input is the block of grid rows
[t-167, t] of the shared feature columns (every feature at row t only uses information up to
t-1 for traffic, the calendar of t and the weather observed at t, as in the problem definition).
`weather_main` enters as an integer code for an embedding (vocabulary of the training rows,
unknown token for the rest).

Statistics follow the group dl.py exactly: they are computed over all cells of the training
windows (a grid row counts once per training window that contains it). Missing lag cells are
imputed with that training median, then every numeric channel is standardised with the mean and
standard deviation (+1e-6) of the imputed training windows; the target with the training mean and
standard deviation. The windows themselves are never materialised: grid rows are weighted by the
number of training windows that cover them, which gives the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LOOKBACK, TARGET

SPLITS = ("train", "val", "test")


@dataclass
class WindowData:
    numeric: list[str]
    use_cat: bool
    X: np.ndarray                 # (T, F) float32, imputed + standardised, full grid
    C: np.ndarray                 # (T,) int64 weather code (0 when use_cat is False)
    y_grid: np.ndarray            # (T,) float32 standardised target on the grid (0 where missing)
    pos: dict                     # split -> grid positions (hour_index) of the target hours
    y: dict                       # split -> targets in vehicles/hour
    times: dict                   # split -> timestamps of the target hours
    y_mean: float
    y_std: float
    vocab: dict = field(default_factory=dict)
    unk: int = 0
    lookback: int = LOOKBACK
    nan_imputed: int = 0
    num_mean: np.ndarray | None = None
    num_std: np.ndarray | None = None

    @property
    def n_cat(self) -> int:
        return len(self.vocab) + 1 if self.use_cat else 0

    @property
    def n_features(self) -> int:
        return len(self.numeric) + (1 if self.use_cat else 0)


def window_counts(pos: np.ndarray, T: int, lookback: int) -> np.ndarray:
    """For every grid row, the number of windows (ending at `pos`) that contain it."""
    diff = np.zeros(T + 1, dtype=np.int64)
    np.add.at(diff, pos - lookback + 1, 1)
    np.add.at(diff, pos + 1, -1)
    return np.cumsum(diff)[:T]


def weighted_nanmedian(x: np.ndarray, w: np.ndarray) -> float:
    """np.nanmedian of the values of `x` repeated `w` times (exact, without repeating)."""
    ok = ~np.isnan(x) & (w > 0)
    if not ok.any():
        return float("nan")
    order = np.argsort(x[ok], kind="stable")
    v, cw = x[ok][order], np.cumsum(w[ok][order])
    m = int(cw[-1])

    def kth(k):                     # k-th value (0-based) of the expanded sorted array
        return v[np.searchsorted(cw, k, side="right")]

    return float(kth((m - 1) // 2)) if m % 2 else float((kth(m // 2 - 1) + kth(m // 2)) / 2)


def build_windows(frame: pd.DataFrame, grid: pd.DataFrame, numeric: list[str],
                  use_cat: bool = True, lookback: int = LOOKBACK) -> WindowData:
    T = len(grid)
    pos = {s: frame.loc[frame["split"] == s, "hour_index"].to_numpy(np.int64) for s in SPLITS}
    y = {s: frame.loc[frame["split"] == s, TARGET].to_numpy(np.float64) for s in SPLITS}
    times = {s: frame.loc[frame["split"] == s, "date_time"].to_numpy() for s in SPLITS}
    if (pos["train"] - lookback + 1).min() < 0:
        raise ValueError("a window starts before the grid")

    X = grid[numeric].to_numpy(np.float64)
    w_all = window_counts(np.concatenate([pos[s] for s in SPLITS]), T, lookback)
    nan_imputed = int((np.isnan(X) * w_all[:, None]).sum())        # missing cells over all windows
    w = window_counts(pos["train"], T, lookback).astype(np.float64)  # train-window multiplicity
    med = np.array([weighted_nanmedian(X[:, j], w) for j in range(X.shape[1])])
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), med, X)
    mean = (X * w[:, None]).sum(0) / w.sum()
    std = np.sqrt(((X - mean) ** 2 * w[:, None]).sum(0) / w.sum()) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    if use_cat:
        train_cats = sorted(frame.loc[frame["split"] == "train", "weather_main"].astype(str).unique())
        vocab = {c: i for i, c in enumerate(train_cats)}
        unk = len(vocab)
        C = grid["weather_main"].astype(str).map(vocab).fillna(unk).astype(np.int64).to_numpy()
    else:
        vocab, unk, C = {}, 0, np.zeros(T, dtype=np.int64)

    y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
    yg = grid[TARGET].to_numpy(np.float64)
    y_grid = np.where(np.isnan(yg), 0.0, (yg - y_mean) / y_std).astype(np.float32)
    return WindowData(numeric=list(numeric), use_cat=use_cat, X=X, C=C, y_grid=y_grid, pos=pos, y=y,
                      times=times, y_mean=y_mean, y_std=y_std, vocab=vocab, unk=unk,
                      lookback=lookback, nan_imputed=nan_imputed, num_mean=mean, num_std=std)
