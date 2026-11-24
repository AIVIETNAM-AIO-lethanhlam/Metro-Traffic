"""End-to-end experiment, split into steps so the notebook can run and inspect each one.

    exp = Experiment(Config())
    exp.load_data(); exp.run_baselines(); exp.train_default_models()
    exp.tune(); exp.select_best(); exp.ablation(); exp.explain(); exp.save()

or simply `exp.run_all()`.
"""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import pandas as pd

from . import artifacts
from .baselines import fit_ridge, naive_forecast, ridge_predict
from .config import Config
from .data import data_quality_report, load_raw, preprocess
from .features import SEASON_LAG, build_dataset
from .metrics import ResultsTracker


def report_name(key: str) -> str:
    """'LSTM' -> 'DL-LSTM', 'LSTM-tuned' -> 'DL-LSTM (tuned)'."""
    return f"DL-{key[:-6]} (tuned)" if key.endswith("-tuned") else f"DL-{key}"


class Experiment:
    def __init__(self, cfg: Config | None = None, verbose: bool = True):
        self.cfg = cfg or Config()
        self.verbose = verbose
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        self.models, self.histories, self.val_scores, self.specs = {}, {}, {}, {}
        self.tune_log = pd.DataFrame()
        self.best: dict | None = None
        self.perm_importance = self.ig_lag = self.ig_features = self.ig_heatmap = None
        self._device = self._batcher = None

    def log(self, *a):
        if self.verbose:
            print(*a)

    # ------------------------------------------------------------------ data
    def load_data(self):
        cfg = self.cfg
        self.raw = load_raw(cfg.data_path)
        self.quality = data_quality_report(self.raw)
        self.hourly, self.wx_cols = preprocess(self.raw, cfg.max_gap)
        self.ds = build_dataset(self.hourly, self.wx_cols, cfg)
        ds = self.ds
        self.y_val = ds.targets(ds.idx_val)
        self.tracker = ResultsTracker(ds.targets(ds.idx_test), naive_forecast(ds, ds.idx_test, SEASON_LAG))
        self.log(f"hourly grid: {len(self.hourly):,} rows | valid hours: {int(ds.valid.sum()):,} | "
                 f"{len(ds.past_cols)} past features, {len(ds.fut_cols)} future features")
        self.log(ds.split_summary().to_string(index=False))
        return self.ds

    # ------------------------------------------------------------- baselines
    def run_baselines(self):
        ds, tr = self.ds, self.tracker
        tr.report("Daily-naive", naive_forecast(ds, ds.idx_test, 24), verbose=self.verbose)
        tr.report("Seasonal-naive (weekly)", naive_forecast(ds, ds.idx_test, SEASON_LAG), verbose=self.verbose)
        self.ridge, alpha, val_mae = fit_ridge(ds, verbose=self.verbose)
        tr.report("Ridge (linear, lags)", ridge_predict(ds, self.ridge, ds.idx_test),
                  dict(val_MAE=val_mae, params=self.ridge.coef_.size), verbose=self.verbose)

    # ----------------------------------------------------------------- torch
    def _torch(self):
        if self._device is None:
            from .training import WindowBatcher, get_device
            self._device = get_device()
            self._batcher = WindowBatcher(self.ds, self._device)
            self.log(f"device: {self._device}")
        return self._device, self._batcher

    def default_spec(self, kind: str) -> dict:
        from .models import DEFAULT_HP
        c = self.cfg
        return dict(kind=kind, hp=copy.deepcopy(DEFAULT_HP[kind]), lr=c.lr, wd=c.weight_decay,
                    batch_size=c.batch_size, head_hidden=256)

    def fit_spec(self, spec: dict, epochs=None, patience=None, batcher=None, verbose=None):
        from .models import build_model, n_params
        from .training import seed_everything, train_model
        device, default_batcher = self._torch()
        batcher = batcher or default_batcher
        seed_everything(self.cfg.seed)
        model = build_model(spec["kind"], batcher.n_past, batcher.n_fut, self.ds.lookback, self.ds.horizon,
                            device=device, head_hidden=spec["head_hidden"], **copy.deepcopy(spec["hp"]))
        model, hist, best_val, secs = train_model(
            model, batcher, self.ds.idx_train, self.ds.idx_val, self.y_val,
            epochs=epochs or self.cfg.epochs, lr=spec["lr"], wd=spec["wd"], batch_size=spec["batch_size"],
            patience=patience or self.cfg.patience, verbose=self.verbose if verbose is None else verbose)
        return model, hist, best_val, secs, n_params(model)

    def _register(self, key: str, spec: dict, model, hist, best_val, secs, n_par):
        from .training import predict
        self.models[key], self.histories[key], self.val_scores[key], self.specs[key] = model, hist, best_val, spec
        _, batcher = self._torch()
        self.tracker.report(report_name(key), predict(model, batcher, self.ds.idx_test),
                            dict(val_MAE=best_val, params=n_par, train_sec=secs), verbose=self.verbose)

    # ------------------------------------------------------ default DL runs
    def train_default_models(self):
        for kind in self.cfg.models:
            self.log(f"\n===== {kind} =====")
            spec = self.default_spec(kind)
            model, hist, v, secs, npar = self.fit_spec(spec)
            self.log(f"-> best val MAE {v:.1f} | {secs:.0f}s | {npar:,} parameters")
            self._register(kind, spec, model, hist, v, secs, npar)

    # ------------------------------------------------------------- tuning
    def tune(self):
        """Random search on the validation set for the `tune_top_k` best default architectures,
        then retrain the best configuration found with the full epoch budget."""
        from .models import COMMON_SPACE, SEARCH_SPACES
        cfg = self.cfg
        if not cfg.run_tuning:
            self.log("tuning disabled (cfg.run_tuning = False)")
            return
        defaults = {k: v for k, v in self.val_scores.items() if not k.endswith("-tuned")}
        top = sorted(defaults, key=defaults.get)[:cfg.tune_top_k]
        self.log(f"tuning: {top}")
        rng = random.Random(cfg.seed)
        rows, best_found = [], {}
        for kind in top:
            best_found[kind] = (defaults[kind], None)
            for trial in range(cfg.n_trials):
                hp = {k: rng.choice(v) for k, v in SEARCH_SPACES[kind].items()}
                cm = {k: rng.choice(v) for k, v in COMMON_SPACE.items()}
                spec = dict(kind=kind, hp=hp, **cm)
                _, _, v, secs, npar = self.fit_spec(spec, epochs=cfg.tune_epochs, patience=4, verbose=False)
                rows.append(dict(model=kind, trial=trial, val_MAE=v, secs=secs, params=npar, **hp, **cm))
                self.log(f"{kind:17s} trial {trial} | val MAE {v:7.1f} | {secs:4.0f}s | {hp} {cm}")
                if v < best_found[kind][0]:
                    best_found[kind] = (v, spec)
        self.tune_log = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)

        for kind, (v, spec) in best_found.items():
            if spec is None:
                self.log(f"{kind}: no trial beat the default configuration -> keep default")
                continue
            self.log(f"\n===== {kind} (tuned) : {spec} =====")
            model, hist, v2, secs, npar = self.fit_spec(spec, verbose=False)
            self.log(f"-> val MAE {v2:.1f} ({secs:.0f}s)")
            self._register(f"{kind}-tuned", spec, model, hist, v2, secs, npar)

    # ----------------------------------------------------------- selection
    def select_best(self):
        """The final model is chosen on VALIDATION MAE (never on the test set)."""
        key = min(self.val_scores, key=self.val_scores.get)
        self.best = dict(key=key, name=report_name(key), val_MAE=self.val_scores[key], spec=self.specs[key])
        self.log(f"best model by validation: {self.best['name']} (val MAE {self.best['val_MAE']:.1f})")
        return self.best

    @property
    def best_model(self):
        return self.models[self.best["key"]]

    # ------------------------------------------------------------ ablation
    def ablation(self):
        """Retrain the best configuration without any weather feature."""
        if not self.cfg.run_ablation:
            self.log("ablation disabled (cfg.run_ablation = False)")
            return
        from .training import WindowBatcher, predict
        device, _ = self._torch()
        cols = [c for c in self.ds.past_cols if c not in self.ds.weather_feats]
        batcher_nw = WindowBatcher(self.ds, device, past_cols=cols)
        model, hist, v, secs, npar = self.fit_spec(self.best["spec"], batcher=batcher_nw, verbose=False)
        name = f"Ablation: {self.best['name']} without weather"
        self.histories["ablation-no-weather"] = hist
        self.tracker.report(name, predict(model, batcher_nw, self.ds.idx_test),
                            dict(val_MAE=v, params=npar, train_sec=secs), verbose=self.verbose)

    # ------------------------------------------------------- explanations
    def explain(self):
        from .explain import ig_summary, permutation_importance
        _, batcher = self._torch()
        self.log("permutation importance ...")
        self.perm_importance = permutation_importance(self.best_model, batcher, self.ds, self.ds.idx_test)
        self.log("integrated gradients ...")
        self.ig_lag, self.ig_features, self.ig_heatmap = ig_summary(
            self.best_model, batcher, self.ds, self.ds.idx_test, n_samples=self.cfg.ig_samples, seed=self.cfg.seed)

    # ---------------------------------------------------------------- save
    def save(self):
        artifacts.save_experiment(self)
        self.log(f"artifacts written to {os.path.abspath(self.cfg.out_dir)}")

    def run_all(self):
        self.load_data()
        self.run_baselines()
        self.train_default_models()
        self.tune()
        self.select_best()
        self.ablation()
        self.explain()
        self.save()
        return self.tracker.table()
