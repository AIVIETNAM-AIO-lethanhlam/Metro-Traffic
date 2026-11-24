"""Optuna tuning of one architecture on validation MAE (group dl.py `suggest` / `objective`):
TPE sampler (seed 42), 10 trials, at most 25 epochs with patience 5, seed 42 for every trial.
Without Optuna installed, random search over the same space is used instead.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import SEARCH_SPACE, build_model
from .training import WindowBatcher, set_seed, train_model

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:                     # pragma: no cover
    optuna = None
    HAS_OPTUNA = False


def _suggest_optuna(trial) -> dict:
    hp = {}
    for k, v in SEARCH_SPACE.items():
        if isinstance(v, list):
            hp[k] = trial.suggest_categorical(k, v)
        elif len(v) == 3 and v[2] == "int":
            hp[k] = trial.suggest_int(k, v[0], v[1])
        else:
            hp[k] = trial.suggest_float(k, v[0], v[1], log=len(v) == 3 and v[2] == "log")
    return hp


def _suggest_random(rng: np.random.Generator) -> dict:
    hp = {}
    for k, v in SEARCH_SPACE.items():
        if isinstance(v, list):
            hp[k] = v[int(rng.integers(len(v)))]
        elif len(v) == 3 and v[2] == "int":
            hp[k] = int(rng.integers(v[0], v[1] + 1))
        elif len(v) == 3 and v[2] == "log":
            hp[k] = float(math.exp(rng.uniform(math.log(v[0]), math.log(v[1]))))
        else:
            hp[k] = float(rng.uniform(v[0], v[1]))
    return hp


def _py(v):
    return v.item() if hasattr(v, "item") else v


def tune(name: str, batcher: WindowBatcher, pos_train, pos_val, y_val, n_trials: int = 10,
         max_epochs: int = 25, patience: int = 5, seed: int = 42, verbose: bool = True):
    """Returns (best config, best val MAE, trial log DataFrame)."""
    rows = []

    def objective(hp: dict) -> float:
        cfg = {**hp, "seed": seed}
        set_seed(seed)
        model = build_model(name, cfg, batcher.n_num, batcher.n_cat, batcher.use_cat, batcher.device)
        _, hist, best, secs = train_model(model, batcher, pos_train, pos_val, y_val, cfg,
                                          max_epochs=max_epochs, patience=patience)
        rows.append(dict(model=name, trial=len(rows), val_MAE=best, epochs=len(hist), seconds=secs, **hp))
        if verbose:
            print(f"    trial {len(rows) - 1:2d}: val MAE {best:7.1f} ({secs:.0f}s) {hp}")
        return best

    if HAS_OPTUNA:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda t: objective(_suggest_optuna(t)), n_trials=n_trials)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            objective(_suggest_random(rng))
    log = pd.DataFrame(rows)
    best_row = log.loc[log["val_MAE"].idxmin()]
    best = {k: _py(best_row[k]) for k in SEARCH_SPACE}
    for k in ("hidden", "layers", "batch", "emb"):
        best[k] = int(best[k])
    return best, float(best_row["val_MAE"]), log
