"""Feature engineering, train-only scaling, chronological split and sliding windows."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config

CAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "is_weekend", "is_holiday", "is_offday"]
SCALE_COLS = ["traffic", "temp_c", "rain_log", "snow_log"]
SEASON_LAG = 168   # weekly seasonality, used by the seasonal-naive baseline and the model skip


def add_features(hourly: pd.DataFrame, wx_cols: list[str]) -> pd.DataFrame:
    """Build the model features from the cleaned hourly frame (unscaled)."""
    f = pd.DataFrame(index=hourly.index)
    f["traffic"] = hourly["traffic_volume"]
    f["temp_c"] = hourly["temp"] - 273.15
    f["rain_log"] = np.log1p(hourly["rain_1h"])          # rain is extremely skewed
    f["snow_log"] = np.log1p(hourly["snow_1h"] * 10)
    f["clouds"] = hourly["clouds_all"] / 100.0
    for c in wx_cols:
        f[c] = hourly[c]
    idx = hourly.index
    # cyclical encodings: hour 23 and hour 0 end up close to each other
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    f["is_holiday"] = hourly["is_holiday"]
    f["is_offday"] = ((f["is_weekend"] + f["is_holiday"]) > 0).astype(float)
    f["is_imputed"] = hourly["is_imputed"]
    f["valid"] = hourly["valid"]
    return f


def day_type(is_holiday, is_weekend) -> np.ndarray:
    return np.select([np.asarray(is_holiday) == 1, np.asarray(is_weekend) == 1],
                     ["Holiday", "Weekend"], "Weekday")


@dataclass
class Dataset:
    """Everything the models need, as dense arrays over the full hourly grid."""
    feat: pd.DataFrame                  # unscaled features
    times: pd.DatetimeIndex
    X_past: np.ndarray                  # (T, n_past) float32, scaled
    X_fut: np.ndarray                   # (T, n_fut) float32, known-future calendar
    y_raw: np.ndarray                   # (T,) vehicles/hour
    y_sc: np.ndarray                    # (T,) scaled target
    valid: np.ndarray                   # (T,) bool
    past_cols: list[str]
    fut_cols: list[str]
    weather_feats: list[str]
    wx_cols: list[str]
    y_mean: float
    y_std: float
    lookback: int
    horizon: int
    idx_train: np.ndarray               # forecast origins (row positions) per split
    idx_val: np.ndarray
    idx_test: np.ndarray
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    scalers: dict = field(default_factory=dict)

    @property
    def past_off(self) -> np.ndarray:
        return np.arange(-self.lookback + 1, 1)

    @property
    def fut_off(self) -> np.ndarray:
        return np.arange(1, self.horizon + 1)

    def target_pos(self, ix) -> np.ndarray:
        return np.asarray(ix)[:, None] + self.fut_off

    def targets(self, ix) -> np.ndarray:
        """(N, H) true values in vehicles/hour."""
        return self.y_raw[self.target_pos(ix)]

    def lagged(self, ix, lag: int) -> np.ndarray:
        """(N, H) value observed `lag` hours before each target hour."""
        return self.y_raw[self.target_pos(ix) - lag]

    def target_times(self, ix) -> np.ndarray:
        return self.times.values[self.target_pos(ix)]

    def split_summary(self) -> pd.DataFrame:
        rows = []
        for name, ix in [("train", self.idx_train), ("val", self.idx_val), ("test", self.idx_test)]:
            rows.append(dict(split=name, windows=len(ix),
                             first_target=str(self.times[ix[0] + 1]),
                             last_target=str(self.times[ix[-1] + self.horizon])))
        return pd.DataFrame(rows)


def build_dataset(hourly: pd.DataFrame, wx_cols: list[str], cfg: Config) -> Dataset:
    if cfg.lookback < SEASON_LAG:
        raise ValueError(f"lookback must be >= {SEASON_LAG} (weekly seasonal skip)")
    feat = add_features(hourly, wx_cols)
    weather_feats = ["temp_c", "rain_log", "snow_log", "clouds"] + wx_cols
    past_cols = ["traffic_scaled"] + weather_feats + CAL_COLS + ["is_imputed"]
    fut_cols = list(CAL_COLS)

    train_end, val_end = pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.val_end)
    train_rows = (feat.index <= train_end) & feat["valid"]

    # scaling is fitted on training rows only (no information from val/test)
    fs = feat.copy()
    scalers = {}
    for c in SCALE_COLS:
        sc = StandardScaler().fit(feat.loc[train_rows, [c]])
        scalers[c] = sc
        fs["traffic_scaled" if c == "traffic" else c] = sc.transform(feat[[c]]).ravel()

    valid = feat["valid"].to_numpy(bool)
    times = feat.index
    L, H, T = cfg.lookback, cfg.horizon, len(times)

    # a window [t-L+1, t+H] is usable only if no hour falls inside a long gap
    inv_cum = np.concatenate([[0], np.cumsum(~valid)])
    origins = np.arange(L - 1, T - H)
    bad = inv_cum[origins + H + 1] - inv_cum[origins - L + 1]
    origins = origins[bad == 0]

    # the split is decided by the target range (inputs may reach back into the previous split)
    tgt_start, tgt_end = times[origins + 1], times[origins + H]
    idx_train = origins[tgt_end <= train_end]
    idx_val = origins[(tgt_start > train_end) & (tgt_end <= val_end)]
    idx_test = origins[tgt_start > val_end]

    return Dataset(
        feat=feat, times=times,
        X_past=fs[past_cols].fillna(0).to_numpy(np.float32),
        X_fut=fs[fut_cols].fillna(0).to_numpy(np.float32),
        y_raw=feat["traffic"].to_numpy(np.float64),
        y_sc=fs["traffic_scaled"].fillna(0).to_numpy(np.float32),
        valid=valid, past_cols=past_cols, fut_cols=fut_cols,
        weather_feats=weather_feats, wx_cols=wx_cols,
        y_mean=float(scalers["traffic"].mean_[0]), y_std=float(scalers["traffic"].scale_[0]),
        lookback=L, horizon=H, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        train_end=train_end, val_end=val_end, scalers=scalers)
