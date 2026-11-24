"""Data cleaning and feature engineering of the shared one-hour-ahead pipeline.

`clean`, `add_features`, `split_frame`, `add_how_profile` and `feature_columns` reproduce the logic
of the group pipeline (`p4aids-quickeda/src/traffic_forecast/data.py`, commit e9a8529) line by
line, so that the deep learning models use exactly the same rows, splits and feature columns as
the ML models. When the group's processed files are copied into `data/processed/`, they are used
directly and nothing is re-derived (shared data contract).

Summary of the shared pipeline
- duplicate timestamps (7,629 rows) collapsed, keeping the first row;
- reindex to an hourly grid with an `observed` mask; the target is never imputed;
- one implausible rain record (> 200 mm/h) set to missing;
- calendar, Fourier, weather, gap-aware lags and rolling statistics, holiday flags spread to the
  whole day (+ day before / after), robust weekly profile and hour-of-week mean (train only);
- rows from 2016-01-01 with an observed target, a fully observed 168-hour history and complete
  weather; chronological 70/15/15 split of the covered time span.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .config import (LAGS, LOOKBACK, ROLL_WINDOWS, START_DATE, TARGET, TIME_COL, TRAIN_FRAC,
                     UCI_URL, VAL_FRAC, WEEKS_BACK, Config)

HOLIDAY_FEATURES = ["pre_holiday", "post_holiday"]
WEEKLY_FEATURES = ["wk_median4", "wk_step_med", "wk_spread", "lag1_vs_med"]
HOW_FEATURES = ["hour_of_week", "how_mean", "lag1_minus_how"]
ENGINEERED_FEATURES = HOLIDAY_FEATURES + WEEKLY_FEATURES + HOW_FEATURES
LEAKY_FEATURES = ["how_mean", "lag1_minus_how"]   # train-wide target mean, see post-merge plan 4.1
WEATHER_FEATURES = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "is_rain", "is_snow"]
CATEGORICAL = ["weather_main"]

MODEL_FRAME = "metro_traffic_model_frame"
FULL_GRID = "metro_traffic_full_grid"
META = "metro_traffic_meta.json"


# ----------------------------------------------------------------------------- raw data
def load_raw(path: str, download_if_missing: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        if not download_if_missing:
            raise FileNotFoundError(path)
        print(f"{path} not found -> downloading from UCI ...")
        buf = io.BytesIO(urllib.request.urlopen(UCI_URL).read())
        with zipfile.ZipFile(buf) as z:
            name = [n for n in z.namelist() if n.endswith((".csv.gz", ".csv"))][0]
            payload = z.read(name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    return df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(df)
    dup_ts = int(df[TIME_COL].duplicated().sum())
    df = df.drop_duplicates(TIME_COL, keep="first").set_index(TIME_COL)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df.index.name = TIME_COL

    df["observed"] = df[TARGET].notna().astype("int8")
    df["temp_c"] = df["temp"] - 273.15
    df["is_holiday"] = df["holiday"].notna().astype("int8")
    df["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)
    df["is_rain"] = (df["rain_1h"].fillna(0) > 0).astype("int8")
    df["is_snow"] = (df["snow_1h"].fillna(0) > 0).astype("int8")
    implausible = int((df["rain_1h"] > 200).sum())
    df.loc[df["rain_1h"] > 200, "rain_1h"] = np.nan

    audit = {
        "raw_rows": raw_rows,
        "duplicate_timestamps": dup_ts,
        "unique_hours": int(df["observed"].sum()),
        "grid_hours": len(df),
        "grid_completeness_pct": round(100 * float(df["observed"].mean()), 2),
        "first": str(df.index.min()),
        "last": str(df.index.max()),
        "missing_target_pct": round(100 * float(df[TARGET].isna().mean()), 2),
        "missing_holiday_pct": round(100 * float(df["holiday"].isna().mean()), 2),
        "implausible_rain_rows": implausible,
    }
    return df, audit


def _gap_aware_lags(s: pd.Series, observed: pd.Series, lag: int) -> pd.Series:
    valid = observed.shift(1).rolling(lag, min_periods=lag).sum().eq(lag)
    return s.shift(lag).where(valid)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dow"] = idx.day_of_week
    out["month"] = idx.month
    out["doy"] = idx.day_of_year
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    for k in (1, 2):
        out[f"hour_sin{k}"] = np.sin(2 * np.pi * k * out["hour"] / 24)
        out[f"hour_cos{k}"] = np.cos(2 * np.pi * k * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)

    target = out[TARGET]
    obs = out["observed"]
    for lag in LAGS:
        out[f"lag_{lag}"] = _gap_aware_lags(target, obs, lag)
    shifted = target.shift(1)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}"] = shifted.rolling(w, min_periods=w).std()
    out["diff_1"] = out["lag_1"] - out["lag_2"]

    day = pd.Series(idx.normalize(), index=idx)
    hol_days = pd.DatetimeIndex(idx[out["is_holiday"].eq(1)].normalize().unique())
    out["is_holiday"] = day.isin(hol_days).astype("int8")
    out["pre_holiday"] = (day + pd.Timedelta(days=1)).isin(hol_days).astype("int8")
    out["post_holiday"] = (day - pd.Timedelta(days=1)).isin(hol_days).astype("int8")

    def weeks(offset: int) -> pd.DataFrame:
        return pd.concat([target.shift(168 * k + offset) for k in WEEKS_BACK], axis=1)

    same_hour = weeks(0)
    out["wk_median4"] = same_hour.median(axis=1)
    out["wk_spread"] = same_hour.max(axis=1) - same_hour.min(axis=1)
    out["wk_step_med"] = (same_hour - weeks(1).to_numpy()).median(axis=1)
    out["lag1_vs_med"] = target.shift(1) - weeks(1).median(axis=1)
    out["hour_of_week"] = out["dow"] * 24 + out["hour"]

    out["history_valid"] = (
        obs.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).sum().eq(LOOKBACK).astype("int8"))
    out["hour_index"] = np.arange(len(out), dtype=np.int64)
    return out


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.Series]:
    df = df[df.index >= pd.Timestamp(START_DATE)]
    df = df[df["observed"].eq(1) & df["history_valid"].eq(1)].copy()
    df = df.dropna(subset=["temp_c", "rain_1h", "snow_1h", "clouds_all"])

    n = len(df)
    t0, t1 = df.index.min(), df.index.max()
    train_end = t0 + (t1 - t0) * TRAIN_FRAC
    val_end = t0 + (t1 - t0) * VAL_FRAC
    df["split"] = np.where(df.index <= train_end, "train", np.where(df.index <= val_end, "val", "test"))
    profile = df.loc[df["split"] == "train"].groupby("hour_of_week")[TARGET].mean()
    add_how_profile(df, profile)

    counts = df["split"].value_counts().reindex(["train", "val", "test"]).to_dict()
    meta = {
        "rows": n,
        "train": int(counts["train"]), "val": int(counts["val"]), "test": int(counts["test"]),
        "train_start": str(df.index[df["split"] == "train"].min()),
        "train_end": str(df.index[df["split"] == "train"].max()),
        "val_end": str(df.index[df["split"] == "val"].max()),
        "test_start": str(df.index[df["split"] == "test"].min()),
        "test_end": str(df.index[df["split"] == "test"].max()),
    }
    return df, meta, profile


def add_how_profile(df: pd.DataFrame, profile: pd.Series) -> None:
    df["how_mean"] = df["hour_of_week"].map(profile)
    df["lag1_minus_how"] = df["lag_1"] - df["how_mean"]


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {TARGET, "observed", "history_valid", "split", "holiday", "weather_description",
               "temp", "hour_index", *CATEGORICAL}
    numeric = [c for c in df.columns if c not in exclude]
    numeric = [c for c in numeric if pd.api.types.is_numeric_dtype(df[c])]
    return numeric, list(CATEGORICAL)


def build(raw_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the model frame, the full hourly grid and the metadata from the raw CSV."""
    cleaned, audit = clean(load_raw(raw_path))
    feats = add_features(cleaned)
    frame, split_meta, profile = split_frame(feats)
    add_how_profile(feats, profile)
    numeric, categorical = feature_columns(frame)
    meta = {"audit": audit, "split": split_meta, "numeric_features": numeric,
            "categorical_features": categorical, "engineered_features": ENGINEERED_FEATURES,
            "target": TARGET, "lookback": LOOKBACK, "seasonal_lag": 24}
    return frame.reset_index(), feats.reset_index(), meta


# ----------------------------------------------------------------------------- loading
def _read_table(directory: str, stem: str):
    for ext, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return reader(path)
    return None


def _write_table(df: pd.DataFrame, directory: str, stem: str) -> str:
    os.makedirs(directory, exist_ok=True)
    try:
        path = os.path.join(directory, stem + ".parquet")
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = os.path.join(directory, stem + ".csv.gz")
        df.to_csv(path, index=False, compression="gzip")
    return path


def load_processed(cfg: Config, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Return (model frame, full grid, meta, source). Uses the group's processed files when present."""
    frame = _read_table(cfg.processed_dir, MODEL_FRAME)
    grid = _read_table(cfg.processed_dir, FULL_GRID)
    meta_path = os.path.join(cfg.processed_dir, META)
    if frame is not None and grid is not None and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        source = f"processed files in {cfg.processed_dir}"
    else:
        frame, grid, meta = build(cfg.raw_path)
        _write_table(frame, cfg.processed_dir, MODEL_FRAME)
        _write_table(grid, cfg.processed_dir, FULL_GRID)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        source = f"built from {cfg.raw_path} (same logic as the group data.py)"
    for d in (frame, grid):
        d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    frame = frame.sort_values(TIME_COL).reset_index(drop=True)
    grid = grid.sort_values(TIME_COL).reset_index(drop=True)
    if not (grid["hour_index"].to_numpy() == np.arange(len(grid))).all():
        raise ValueError("full grid must be a complete hourly index")
    if verbose:
        s = meta["split"]
        print(f"data: {source}\n  rows {s['rows']:,} = train {s['train']:,} / val {s['val']:,} / "
              f"test {s['test']:,}; {len(meta['numeric_features'])} numeric + "
              f"{len(meta['categorical_features'])} categorical features")
    return frame, grid, meta, source


def feature_groups(numeric: list[str]) -> dict[str, list[str]]:
    """Ablation groups, same rules as the group's `run_ablation` (evaluation.py)."""
    eng = [c for c in ENGINEERED_FEATURES if c in numeric]
    return {
        "calendar": [c for c in numeric if (c == "hour" or c.startswith(
            ("hour_", "dow", "month", "doy", "is_weekend", "is_holiday"))) and c not in eng],
        "weather": [c for c in WEATHER_FEATURES if c in numeric],
        "lags": [c for c in numeric if c.startswith(("lag_", "roll_", "diff_")) and c not in eng],
        "engineered": eng,
    }
