"""Save / load experiment outputs. The dashboard and the export script only read these
files, so they need neither PyTorch, a GPU nor the raw data."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

FILES = dict(
    config="config.json", data="data_meta.json", splits="split_info.csv", results="results.csv",
    seeds="seed_results.csv", screening="screening.csv", tuning="tuning_log.csv", best="best_model.json",
    histories="histories.json", predictions="predictions.npz", ablation="ablation.csv",
    ig_features="ig_features.csv", ig_lag="ig_lag.csv", ig_matrix="ig_matrix.csv",
    occlusion="weather_occlusion.csv", groups="group_importance.csv", dm="dm_tests.csv",
    hourly="hourly.csv.gz",
)


def _p(out_dir, key):
    return os.path.join(out_dir, FILES[key])


def _json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _ns(times) -> np.ndarray:
    return pd.to_datetime(np.asarray(times)).values.astype("datetime64[ns]").astype(np.int64)


def save_experiment(exp) -> None:
    out = exp.cfg.out_dir
    os.makedirs(out, exist_ok=True)
    wd, frame = exp.wd, exp.frame

    exp.cfg.save(_p(out, "config"))
    _json(_p(out, "data"), dict(source=exp.source, audit=exp.meta.get("audit", {}), split=exp.meta.get("split", {}),
                                numeric_features=exp.numeric, categorical_features=["weather_main"],
                                weather_vocab=wd.vocab, lookback=wd.lookback, nan_imputed=wd.nan_imputed,
                                mase_scale=exp.scale, timings=exp.timings, device=str(exp.device)))
    rows = []
    for s in ("train", "val", "test"):
        part = frame[frame["split"] == s]
        rows.append(dict(split=s, rows=len(part), start=part["date_time"].min(), end=part["date_time"].max(),
                         mean_volume=part["traffic_volume"].mean()))
    pd.DataFrame(rows).to_csv(_p(out, "splits"), index=False)

    exp.results_table().to_csv(_p(out, "results"), index=False)
    if exp.seed_rows:
        pd.DataFrame(exp.seed_rows).to_csv(_p(out, "seeds"), index=False)
    for key, attr in [("screening", "screening"), ("tuning", "tune_log"), ("ablation", "ablation"),
                      ("ig_features", "ig_features"), ("ig_lag", "ig_lag"), ("occlusion", "occlusion"),
                      ("groups", "group_importance"), ("dm", "dm")]:
        df = getattr(exp, attr, None)
        if df is not None and len(df):
            df.to_csv(_p(out, key), index=False)
    if exp.ig_matrix is not None:
        exp.ig_matrix.to_csv(_p(out, "ig_matrix"))
    _json(_p(out, "histories"), exp.histories)

    arrays = {}
    for split in ("val", "test"):
        names = list(exp.preds[split])
        part = frame[frame["split"] == split]
        arrays[f"{split}_times"] = _ns(wd.times[split])
        arrays[f"{split}_y"] = wd.y[split].astype(np.float32)
        arrays[f"{split}_names"] = np.array(names, dtype="U")
        arrays[f"{split}_preds"] = np.stack([exp.preds[split][n] for n in names]).astype(np.float32)
        arrays[f"{split}_is_holiday"] = part["is_holiday"].to_numpy(np.int8)
        arrays[f"{split}_is_weekend"] = part["is_weekend"].to_numpy(np.int8)
        arrays[f"{split}_weather"] = part["weather_main"].astype(str).to_numpy().astype("U")
    if exp.best is not None:
        seeds = exp.final[exp.best["model"]]["seed_test"]
        arrays["seed_names"] = np.array([f"seed {s}" for s in seeds])
        arrays["seed_preds"] = np.stack(list(seeds.values())).astype(np.float32)
    np.savez_compressed(_p(out, "predictions"), **arrays)

    if exp.best is not None:
        best = dict(exp.best)
        best["weights"] = []
        try:
            import torch
            for seed, m in zip(exp.cfg.seeds, exp.final[best["model"]].get("ft", [])):
                fn = f"best_model_seed{seed}.pt"
                torch.save(m.state_dict(), os.path.join(out, fn))
                best["weights"].append(fn)
        except Exception as e:                       # weights are optional for the dashboard
            print("could not save weights:", e)
        _json(_p(out, "best"), best)

    g = exp.grid
    split = pd.Series("", index=g.index)
    split.loc[frame["hour_index"].to_numpy()] = frame["split"].to_numpy()
    hourly = pd.DataFrame({
        "date_time": g["date_time"], "traffic_volume": g["traffic_volume"], "observed": g["observed"].astype(int),
        "temp_c": g["temp_c"], "rain_1h": g["rain_1h"], "snow_1h": g["snow_1h"], "clouds_all": g["clouds_all"],
        "weather_main": g["weather_main"], "is_holiday": g["is_holiday"], "is_weekend": g["is_weekend"],
        "split": split.to_numpy()})
    hourly.to_csv(_p(out, "hourly"), index=False, compression="gzip")


def day_type(is_holiday, is_weekend) -> np.ndarray:
    return np.select([np.asarray(is_holiday) == 1, np.asarray(is_weekend) == 1], ["Holiday", "Weekend"], "Weekday")


def load_artifacts(out_dir: str) -> dict:
    """Everything that exists in `out_dir`; missing files map to None."""
    def read(key, fn):
        path = _p(out_dir, key)
        return fn(path) if os.path.exists(path) else None

    def rjson(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    A = {k: read(k, pd.read_csv) for k in ("splits", "results", "seeds", "screening", "tuning", "ablation",
                                            "ig_features", "ig_lag", "occlusion", "groups", "dm")}
    A["config"] = read("config", rjson)
    A["data"] = read("data", rjson)
    A["best"] = read("best", rjson)
    A["histories"] = read("histories", lambda p: {k: pd.DataFrame(v) for k, v in rjson(p).items()})
    A["ig_matrix"] = read("ig_matrix", lambda p: pd.read_csv(p, index_col=0))
    A["hourly"] = read("hourly", lambda p: pd.read_csv(p, parse_dates=["date_time"]))

    npz = read("predictions", lambda p: dict(np.load(p, allow_pickle=False)))
    A["predictions"] = None
    if npz is not None:
        P = {}
        for s in ("val", "test"):
            P[s] = dict(times=pd.to_datetime(npz[f"{s}_times"]), y=npz[f"{s}_y"].astype(float),
                        preds={str(n): npz[f"{s}_preds"][i].astype(float) for i, n in enumerate(npz[f"{s}_names"])},
                        day_type=day_type(npz[f"{s}_is_holiday"], npz[f"{s}_is_weekend"]),
                        weather=npz[f"{s}_weather"].astype(str))
        if "seed_preds" in npz:
            P["seeds"] = {str(n): npz["seed_preds"][i].astype(float) for i, n in enumerate(npz["seed_names"])}
        A["predictions"] = P
    return A
