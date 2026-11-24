"""Experiment configuration for the shared one-hour-ahead problem.

The data constants (start date, split fractions, lags, rolling windows, look-back) are the ones of
the group pipeline `p4aids-quickeda/src/traffic_forecast/config.py` (commit e9a8529), so that the
deep learning models here see exactly the same rows, splits and feature columns as the ML models.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

UCI_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
DATA_FILE = "Metro_Interstate_Traffic_Volume.csv.gz"

# ---- shared data contract (do not change without changing the group pipeline) ----
TARGET = "traffic_volume"
TIME_COL = "date_time"
LOOKBACK = 168                     # one week of hourly history per window
LAGS = (1, 2, 3, 24, 48, 168)
ROLL_WINDOWS = (3, 6, 24, 48, 168)
WEEKS_BACK = (1, 2, 3, 4)          # robust weekly profile
START_DATE = "2016-01-01"          # dense post-outage segment
TRAIN_FRAC, VAL_FRAC = 0.70, 0.85  # chronological 70/15/15 of the time span
SEASONAL_LAG = 24                  # seasonal-naive baseline and MASE denominator

ALL_MODELS = ("LSTM", "TCN")      # the two DL architectures of docs/methodology.md


@dataclass
class Config:
    # paths
    raw_path: str = f"data/{DATA_FILE}"
    processed_dir: str = "data/processed"   # put the group's processed files here to reuse them
    out_dir: str = "outputs"
    ml_results_dir: str = ""                 # optional: group results/ folder with ml_predictions.parquet

    # models and seeds (group dl.py: --seeds 3 -> 42, 43, 44)
    models: tuple = ALL_MODELS
    seeds: tuple = (42, 43, 44)              # every model is trained with all seeds -> ensemble mean

    # training (group protocol)
    max_epochs: int = 80
    patience: int = 10
    finetune_epochs: int = 8                 # train + val at lr/5, cosine decay

    # tuning: Optuna for EVERY architecture (group dl.py --trials 10)
    run_tuning: bool = True
    n_trials: int = 10
    tune_max_epochs: int = 25
    tune_patience: int = 5

    # extra experiment (post-merge plan 4.3): cumulative feature-group ablation of the best DL model
    run_ablation: bool = True
    ablation_seeds: tuple = (42, 43, 44)

    # explanation (docs/methodology.md §7)
    ig_samples: int = 256
    ig_steps: int = 24
    drop_leaky_features: bool = False        # drop how_mean / lag1_minus_how (see post-merge plan 4.1)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("models", "seeds", "ablation_seeds"):
            d[k] = list(d[k])
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for k in ("models", "seeds", "ablation_seeds"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
