"""Save / load experiment outputs. The Streamlit dashboard only reads these files,
so it does not need PyTorch, a GPU or the raw data."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd



def day_type(is_holiday, is_weekend) -> np.ndarray:
    # same as features.day_type, duplicated so the dashboard does not need scikit-learn
    return np.select([np.asarray(is_holiday) == 1, np.asarray(is_weekend) == 1],
                     ["Holiday", "Weekend"], "Weekday")

FILES = dict(
    config="config.json", quality="data_quality.json", splits="split_info.csv",
    results="results_test.csv", predictions="predictions_test.npz", model_names="model_names.json",
    histories="histories.json", tuning="tuning_log.csv", best="best_model.json", weights="best_model.pt",
    perm="permutation_importance.csv", ig_lag="ig_lag.csv", ig_features="ig_features.csv",
    ig_heatmap="ig_horizon_heatmap.csv", hourly="hourly_clean.csv.gz",
)


def _p(out_dir, key):
    return os.path.join(out_dir, FILES[key])


def _json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_experiment(exp) -> None:
    out, ds, tr = exp.cfg.out_dir, exp.ds, exp.tracker
    os.makedirs(out, exist_ok=True)

    exp.cfg.save(_p(out, "config"))
    _json(_p(out, "quality"), exp.quality)
    ds.split_summary().to_csv(_p(out, "splits"), index=False)
    tr.table().to_csv(_p(out, "results"))

    names = list(tr.preds)
    pos = ds.target_pos(ds.idx_test)
    np.savez_compressed(
        _p(out, "predictions"),
        origins=ds.idx_test,
        target_times=ds.times.values[pos].astype("datetime64[ns]").astype(np.int64),
        y_true=tr.y_true.astype(np.float32),
        preds=np.stack([tr.preds[n] for n in names]).astype(np.float32),
        is_holiday=ds.feat["is_holiday"].to_numpy()[pos].astype(np.int8),
        is_weekend=ds.feat["is_weekend"].to_numpy()[pos].astype(np.int8),
    )
    _json(_p(out, "model_names"), names)
    _json(_p(out, "histories"), {k: v.to_dict(orient="records") for k, v in exp.histories.items()})
    if len(exp.tune_log):
        exp.tune_log.to_csv(_p(out, "tuning"), index=False)
    if exp.best is not None:
        _json(_p(out, "best"), exp.best)
        import torch
        torch.save(exp.best_model.state_dict(), _p(out, "weights"))
    for key, df, idx in [("perm", exp.perm_importance, False), ("ig_lag", exp.ig_lag, False),
                         ("ig_features", exp.ig_features, False), ("ig_heatmap", exp.ig_heatmap, True)]:
        if df is not None:
            df.to_csv(_p(out, key), index=idx)

    h = exp.hourly
    clean = pd.DataFrame({
        "date_time": h.index, "traffic_volume": h["traffic_volume"].values,
        "temp_c": (h["temp"] - 273.15).values, "rain_1h": h["rain_1h"].values, "snow_1h": h["snow_1h"].values,
        "clouds_all": h["clouds_all"].values, "weather_main": h["weather_main"].values,
        "is_holiday": h["is_holiday"].values.astype(int), "holiday_name": h["holiday_name"].values,
        "is_weekend": (h.index.dayofweek >= 5).astype(int), "is_imputed": h["is_imputed"].values.astype(int),
        "valid": h["valid"].values.astype(int)})
    clean.to_csv(_p(out, "hourly"), index=False, compression="gzip")


def load_artifacts(out_dir: str) -> dict:
    """Everything that exists in `out_dir`; missing files map to None."""
    a = {}

    def read(key, fn):
        path = _p(out_dir, key)
        return fn(path) if os.path.exists(path) else None

    def rjson(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    a["config"] = read("config", rjson)
    a["quality"] = read("quality", rjson)
    a["splits"] = read("splits", pd.read_csv)
    a["results"] = read("results", lambda p: pd.read_csv(p, index_col=0))
    a["histories"] = read("histories", lambda p: {k: pd.DataFrame(v) for k, v in rjson(p).items()})
    a["tuning"] = read("tuning", pd.read_csv)
    a["best"] = read("best", rjson)
    a["perm"] = read("perm", pd.read_csv)
    a["ig_lag"] = read("ig_lag", pd.read_csv)
    a["ig_features"] = read("ig_features", pd.read_csv)
    a["ig_heatmap"] = read("ig_heatmap", lambda p: pd.read_csv(p, index_col=0))
    a["hourly"] = read("hourly", lambda p: pd.read_csv(p, parse_dates=["date_time"]))

    names = read("model_names", rjson)
    npz = read("predictions", lambda p: dict(np.load(p)))
    if names is not None and npz is not None:
        a["predictions"] = dict(
            origins=npz["origins"],
            target_times=pd.to_datetime(npz["target_times"].ravel()).values.reshape(npz["target_times"].shape),
            y_true=npz["y_true"], preds={n: npz["preds"][i] for i, n in enumerate(names)},
            day_type=day_type(npz["is_holiday"], npz["is_weekend"]))
    else:
        a["predictions"] = None
    return a
