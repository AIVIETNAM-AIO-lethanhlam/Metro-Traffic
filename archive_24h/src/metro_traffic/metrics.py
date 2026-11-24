"""Evaluation metrics and a small results tracker."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred, y_naive=None) -> dict:
    """MAE, RMSE, sMAPE (%), R2 and rMAE (= MAE / MAE of the seasonal-naive forecast)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    err = y_pred - y_true
    out = dict(
        MAE=float(np.mean(np.abs(err))),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        sMAPE=float(100 * np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))),
        R2=float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    )
    if y_naive is not None:
        out["rMAE"] = out["MAE"] / float(np.mean(np.abs(np.asarray(y_naive) - y_true)))
    return out


class ResultsTracker:
    """Keeps test metrics and test predictions of every model in one place."""

    def __init__(self, y_true: np.ndarray, y_naive: np.ndarray):
        self.y_true = y_true
        self.y_naive = y_naive
        self.results: dict[str, dict] = {}
        self.preds: dict[str, np.ndarray] = {}

    def report(self, name: str, pred: np.ndarray, extra: dict | None = None, verbose: bool = True) -> dict:
        m = compute_metrics(self.y_true, pred, self.y_naive)
        if extra:
            m.update(extra)
        self.results[name] = m
        self.preds[name] = np.clip(np.asarray(pred, np.float32), 0, None)
        if verbose:
            print(f"{name:34s} MAE={m['MAE']:7.1f} | RMSE={m['RMSE']:7.1f} | sMAPE={m['sMAPE']:5.2f}% "
                  f"| R2={m['R2']:.4f} | rMAE={m['rMAE']:.3f}")
        return m

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results).T
        cols = [c for c in ["MAE", "RMSE", "sMAPE", "R2", "rMAE", "val_MAE", "params", "train_sec"] if c in df.columns]
        df = df[cols].sort_values("MAE")
        df.index.name = "model"
        return df
