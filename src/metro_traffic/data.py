"""Loading and cleaning the raw Metro Interstate Traffic Volume data."""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .config import UCI_URL

RARE_WEATHER = {"Smoke": "Haze", "Squall": "Thunderstorm"}
CONTINUOUS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]


def load_raw(path: str, download_if_missing: bool = True) -> pd.DataFrame:
    """Read the raw CSV (.csv or .csv.gz). Downloads it from UCI if the file is missing."""
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
    raw = pd.read_csv(path)
    raw["date_time"] = pd.to_datetime(raw["date_time"])
    return raw


def data_quality_report(raw: pd.DataFrame) -> dict:
    """Re-check the issues found in the EDA report."""
    per_ts = raw.groupby("date_time")["traffic_volume"].agg(["size", "nunique"])
    return {
        "rows": len(raw),
        "start": str(raw.date_time.min()),
        "end": str(raw.date_time.max()),
        "unique_timestamps": int(raw.date_time.nunique()),
        "duplicate_timestamps": int(raw.date_time.duplicated().sum()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "holiday_missing_pct": round(100 * raw.holiday.isna().mean(), 2),
        "holiday_label_hours": sorted(raw.loc[raw.holiday.notna(), "date_time"].dt.hour.unique().tolist()),
        "temp_zero_kelvin": int((raw.temp == 0).sum()),
        "rain_over_100mm": int((raw.rain_1h > 100).sum()),
        "dup_timestamps_with_different_traffic": int((per_ts.query("size > 1")["nunique"] > 1).sum()),
    }


def preprocess(raw: pd.DataFrame, max_gap: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Clean the raw data and put it on a complete hourly grid.

    Steps
    1. drop exact duplicate rows
    2. merge duplicated timestamps: weather labels of the same hour -> multi-hot columns
    3. sensor errors (temp = 0 K, rain_1h > 100 mm) -> NaN, then interpolated
    4. reindex to a full hourly grid; gaps <= max_gap hours are interpolated (flag
       `is_imputed`), longer gaps stay NaN and are marked `valid = False`
    5. the holiday label only exists on the 00:00 row -> spread it to the whole day

    Returns the hourly frame and the list of weather multi-hot columns.
    """
    df = raw.drop_duplicates().copy()
    df["weather_main"] = df["weather_main"].replace(RARE_WEATHER)

    wx = pd.get_dummies(df[["date_time", "weather_main"]], columns=["weather_main"], prefix="wx", dtype=float)
    wx = wx.groupby("date_time").max()

    num = df.groupby("date_time").agg(
        holiday=("holiday", "first"), temp=("temp", "first"), rain_1h=("rain_1h", "first"),
        snow_1h=("snow_1h", "first"), clouds_all=("clouds_all", "first"),
        traffic_volume=("traffic_volume", "first"),
        weather_main=("weather_main", "first"))
    h = num.join(wx)

    h.loc[h.temp < 200, "temp"] = np.nan        # 0 K
    h.loc[h.rain_1h > 100, "rain_1h"] = np.nan  # 9831 mm/h

    holiday_names = h.loc[h.holiday.notna(), "holiday"]
    holiday_map = dict(zip(holiday_names.index.normalize(), holiday_names.values))
    h = h.drop(columns="holiday")

    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(full)
    h.index.name = "date_time"

    na = h["traffic_volume"].isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    short_gap = na & (run_len <= max_gap)
    long_gap = na & ~short_gap

    wx_cols = sorted(c for c in h.columns if c.startswith("wx_"))
    h[CONTINUOUS] = h[CONTINUOUS].interpolate(method="time", limit_area="inside")
    h[wx_cols + ["weather_main"]] = h[wx_cols + ["weather_main"]].ffill()
    h.loc[long_gap, CONTINUOUS + wx_cols + ["weather_main"]] = np.nan

    h["is_imputed"] = short_gap.astype(float)
    h["valid"] = ~long_gap
    day = h.index.normalize()
    h["is_holiday"] = day.isin(list(holiday_map)).astype(float)
    h["holiday_name"] = pd.Series(day, index=h.index).map(holiday_map)
    return h, wx_cols
