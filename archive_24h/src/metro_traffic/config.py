"""Experiment configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

UCI_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
DATA_FILE = "Metro_Interstate_Traffic_Volume.csv.gz"

ALL_MODELS = ("MLP", "LSTM", "GRU", "TCN", "PatchTransformer")


@dataclass
class Config:
    # paths
    data_path: str = f"data/{DATA_FILE}"
    out_dir: str = "outputs"

    # problem definition
    lookback: int = 168        # 1 week of hourly history as input
    horizon: int = 24          # forecast the next 24 hours
    max_gap: int = 3           # gaps <= 3 h are interpolated; longer gaps are excluded
    train_end: str = "2016-12-31 23:00:00"
    val_end: str = "2017-12-31 23:00:00"   # test = 2018-01-01 .. 2018-09-30

    # training
    batch_size: int = 256
    epochs: int = 40
    patience: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    models: tuple = ALL_MODELS

    # hyperparameter search
    run_tuning: bool = True
    tune_top_k: int = 2        # tune the k best architectures from the default run
    n_trials: int = 8
    tune_epochs: int = 15

    # extra experiments
    run_ablation: bool = True  # retrain the best model without weather features
    ig_samples: int = 512      # test windows used for Integrated Gradients

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["models"] = tuple(d.get("models", ALL_MODELS))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
