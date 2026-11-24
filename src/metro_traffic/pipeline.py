"""End-to-end DL experiment following the group pipeline (docs/problem_definition.md, docs/methodology.md).

    1. shared processed data (group files if present, else rebuilt with the same rules)
    2. baselines: persistence, seasonal naive (t-24), hour-of-week mean
    3. Optuna tuning of EVERY architecture (LSTM, TCN): 10 trials, <= 25 epochs, patience 5, seed 42
    4. final training of each architecture with seeds 42, 43, 44 (<= 80 epochs, patience 10)
       -> base ensemble (mean of seeds, train only); each seed model fine-tuned on train + val for
       8 epochs at lr/5 (cosine) -> fine-tuned ensemble (the reported test forecast)
    5. selection of the best DL model by validation MAE of the base ensemble
    6. (post-merge plan 4.3) cumulative feature-group ablation of the selected model
    7. Integrated Gradients + weather occlusion of the selected model
    8. Diebold-Mariano: best ML (by validation MAE, if the group ML predictions are given) vs best DL;
       also best DL vs the other DL model and the baselines
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from . import data as D
from .baselines import BASELINES, baseline_predictions
from .config import TARGET, Config
from .explain import ensemble_predict, ig_report, sample_test_windows, weather_occlusion
from .metrics import all_metrics, diebold_mariano, mase_scale
from .models import DEFAULT_CFG, build_model, n_params
from .training import WindowBatcher, fine_tune, get_device, predict, set_seed, train_model
from .tuning import HAS_OPTUNA, tune
from .windows import build_windows

BASE_SUFFIX = " (no fine-tune)"
ML_PREFIX = "ML: "
ML_META_COLS = {"date_time", "split", TARGET, "lag_1", "lag_24", "hour", "dow", "month",
                "persistence", "seasonal_naive", "hour_of_week_mean"}


class Experiment:
    def __init__(self, cfg: Config | None = None, verbose: bool = True):
        self.cfg = cfg or Config()
        self.verbose = verbose
        self.device = get_device()
        self.timings: dict[str, float] = {}
        self.histories: dict[str, list] = {}
        self.tune_log = pd.DataFrame()
        self.params: dict[str, dict] = {}
        self.tuned_val: dict[str, float] = {}
        self.final: dict[str, dict] = {}
        self.seed_rows: list[dict] = []
        self.preds = {"val": {}, "test": {}}
        self.best: dict | None = None
        self.best_ml: str | None = None
        self.ablation = pd.DataFrame()
        self.ig_features = self.ig_lag = self.ig_matrix = None
        self.occlusion = None
        self.group_importance = None           # not part of the group protocol
        self.dm = pd.DataFrame()
        self.ml_names: list[str] = []

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ data
    def load_data(self):
        t0 = time.time()
        self.frame, self.grid, self.meta, self.source = D.load_processed(self.cfg, verbose=self.verbose)
        numeric = list(self.meta["numeric_features"])
        if self.cfg.drop_leaky_features:
            numeric = [c for c in numeric if c not in D.LEAKY_FEATURES]
        self.numeric = numeric
        self.wd = build_windows(self.frame, self.grid, numeric, use_cat=True)
        self.y = self.wd.y
        tr = self.frame[self.frame["split"] == "train"]
        self.scale = mase_scale(tr[TARGET], tr["lag_24"])
        self.batcher = WindowBatcher(self.wd, self.device)
        self.pos_trainval = np.concatenate([self.wd.pos["train"], self.wd.pos["val"]])
        n_cells = sum(len(p) for p in self.wd.pos.values()) * self.wd.lookback * len(numeric)
        self.log(f"  windows: {len(numeric)} numeric + weather_main ({len(self.wd.vocab)} categories + unk), "
                 f"lookback {self.wd.lookback} h, {self.wd.nan_imputed:,} missing window cells "
                 f"({100 * self.wd.nan_imputed / n_cells:.1f} %) imputed; device {self.device}")
        self.timings["data"] = time.time() - t0

    def run_baselines(self):
        for split in ("val", "test"):
            self.preds[split].update(baseline_predictions(self.frame, split))

    # ------------------------------------------------------------------ helpers
    def _train(self, name, cfg, batcher=None, max_epochs=None, patience=None, tag=None):
        b = batcher or self.batcher
        set_seed(int(cfg["seed"]))
        model = build_model(name, cfg, b.n_num, b.n_cat, b.use_cat, self.device)
        model, hist, best, secs = train_model(model, b, self.wd.pos["train"], self.wd.pos["val"], self.y["val"],
                                              cfg, max_epochs=max_epochs or self.cfg.max_epochs,
                                              patience=patience or self.cfg.patience)
        if tag:
            self.histories[tag] = hist.to_dict(orient="records")
        return model, dict(val_MAE=best, epochs=len(hist), seconds=secs)

    # ------------------------------------------------------------------ tuning
    def run_tuning(self):
        t0 = time.time()
        logs = []
        for name in self.cfg.models:
            if not self.cfg.run_tuning:
                self.params[name] = dict(DEFAULT_CFG)
                continue
            self.log(f"  tuning {name}: {self.cfg.n_trials} trials "
                     f"({'Optuna TPE' if HAS_OPTUNA else 'random search, Optuna not installed'})")
            best, best_val, log = tune(name, self.batcher, self.wd.pos["train"], self.wd.pos["val"], self.y["val"],
                                       n_trials=self.cfg.n_trials, max_epochs=self.cfg.tune_max_epochs,
                                       patience=self.cfg.tune_patience, seed=42, verbose=self.verbose)
            self.params[name], self.tuned_val[name] = best, best_val
            log["best"] = log["val_MAE"].eq(best_val)
            logs.append(log)
            self.log(f"  {name} tuned: val MAE {best_val:.1f} {best}")
        self.tune_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
        self.timings["tuning"] = time.time() - t0

    # ------------------------------------------------------------------ final multi-seed
    def run_final(self):
        t0 = time.time()
        for name in self.cfg.models:
            params = self.params[name]
            base_models, ft_models, seed_test = [], [], {}
            for seed in self.cfg.seeds:
                cfg = {**params, "seed": seed}
                tag = f"{name} | seed {seed}"
                model, info = self._train(name, cfg, tag=tag)
                ft, losses = fine_tune(model, self.batcher, self.pos_trainval, cfg, epochs=self.cfg.finetune_epochs)
                self.histories[tag + " | fine-tune"] = [dict(epoch=i + 1, train_loss=l) for i, l in enumerate(losses)]
                p_val, p_test = predict(model, self.batcher, self.wd.pos["val"]), predict(model, self.batcher, self.wd.pos["test"])
                p_ft = predict(ft, self.batcher, self.wd.pos["test"])
                row = dict(model=name, seed=seed, n_params=n_params(model),
                           val_MAE=float(np.mean(np.abs(p_val - self.y["val"]))),
                           test_MAE_no_finetune=float(np.mean(np.abs(p_test - self.y["test"]))),
                           test_MAE=float(np.mean(np.abs(p_ft - self.y["test"]))),
                           epochs=info["epochs"], seconds=info["seconds"])
                self.seed_rows.append(row)
                base_models.append(model)
                ft_models.append(ft)
                seed_test[seed] = p_ft
                self.log(f"  {name} seed={seed}: base MAE {row['test_MAE_no_finetune']:.1f} -> ft "
                         f"{row['test_MAE']:.1f} ({info['epochs']} epochs, val {row['val_MAE']:.1f})")
            self.preds["val"][name + BASE_SUFFIX] = ensemble_predict(base_models, self.batcher, self.wd.pos["val"])
            self.preds["test"][name + BASE_SUFFIX] = ensemble_predict(base_models, self.batcher, self.wd.pos["test"])
            self.preds["test"][name] = np.mean(list(seed_test.values()), axis=0)
            val_ens = float(np.mean(np.abs(self.preds["val"][name + BASE_SUFFIX] - self.y["val"])))
            self.final[name] = dict(base=base_models, ft=ft_models, seed_test=seed_test, hp=params,
                                    val_MAE_ensemble=val_ens)
            self.log(f"  {name}: ensemble val MAE {val_ens:.1f} | test MAE "
                     f"{np.mean(np.abs(self.preds['test'][name + BASE_SUFFIX] - self.y['test'])):.1f} -> fine-tuned "
                     f"{np.mean(np.abs(self.preds['test'][name] - self.y['test'])):.1f}")
        self.timings["final"] = time.time() - t0

    def select_best(self):
        name = min(self.final, key=lambda k: self.final[k]["val_MAE_ensemble"])
        f = self.final[name]
        self.best = dict(model=name, hp=f["hp"], tuned=self.cfg.run_tuning, seeds=list(self.cfg.seeds),
                         val_MAE_ensemble=f["val_MAE_ensemble"],
                         test=all_metrics(self.y["test"], self.preds["test"][name], self.scale),
                         test_no_finetune=all_metrics(self.y["test"], self.preds["test"][name + BASE_SUFFIX], self.scale),
                         finetune_epochs=self.cfg.finetune_epochs, n_params=n_params(f["base"][0]),
                         selection="lowest validation MAE of the seed ensemble before fine-tuning")
        self.log(f"  selected {name}: val MAE {f['val_MAE_ensemble']:.1f}, test MAE {self.best['test']['MAE']:.1f}")
        for k in self.final:
            if k != name:
                self.final[k].pop("base", None)
                self.final[k].pop("ft", None)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ ablation (plan 4.3)
    def run_ablation(self):
        if not self.cfg.run_ablation:
            return
        t0 = time.time()
        name, params = self.best["model"], self.best["hp"]
        groups = D.feature_groups(self.numeric)
        stages, cols = [], []
        for g, label in [("calendar", "calendar"), ("weather", "+ weather"), ("lags", "+ lags"),
                         ("engineered", "+ engineered (full)")]:
            cols = cols + groups[g]
            stages.append((label, list(cols), g != "calendar"))
        rows = []
        for i, (label, cols_i, use_cat) in enumerate(stages):
            last = i == len(stages) - 1
            b = self.batcher if last else WindowBatcher(self.wd, self.device, columns=cols_i, use_cat=use_cat)
            for seed in self.cfg.ablation_seeds:
                if last and seed in self.cfg.seeds:                     # full stage = final base runs
                    m = self.final[name]["base"][list(self.cfg.seeds).index(seed)]
                else:
                    m, _ = self._train(name, {**params, "seed": seed}, batcher=b)
                rows.append(dict(stage=i + 1, features=label, n_numeric=len(cols_i), weather_main=use_cat, seed=seed,
                                 val_MAE=float(np.mean(np.abs(predict(m, b, self.wd.pos["val"]) - self.y["val"]))),
                                 test_MAE=float(np.mean(np.abs(predict(m, b, self.wd.pos["test"]) - self.y["test"])))))
            r = pd.DataFrame([x for x in rows if x["stage"] == i + 1])
            self.log(f"  ablation {label:22s} val MAE {r['val_MAE'].mean():7.1f} | test MAE {r['test_MAE'].mean():7.1f}")
        self.ablation = pd.DataFrame(rows)
        self.timings["ablation"] = time.time() - t0

    # ------------------------------------------------------------------ explanation
    def run_explain(self):
        t0 = time.time()
        models = self.final[self.best["model"]]["ft"]
        sample = sample_test_windows(self.wd.pos["test"], self.cfg.ig_samples, seed=42)
        self.ig_features, self.ig_lag, mat = ig_report(models, self.batcher, sample, steps=self.cfg.ig_steps)
        self.ig_matrix = pd.DataFrame(mat, columns=self.batcher.columns,
                                      index=pd.Index(np.arange(-mat.shape[0] + 1, 1), name="lag"))
        self.occlusion = weather_occlusion(models, self.batcher, sample, self.wd.vocab)
        self.timings["explain"] = time.time() - t0

    # ------------------------------------------------------------------ ML predictions + DM
    def load_ml_predictions(self):
        """Group ML predictions: results/ml_predictions.parquet (or .csv) with date_time + one column per model."""
        d = self.cfg.ml_results_dir
        if not d or not os.path.isdir(d):
            return
        paths = sorted(glob.glob(os.path.join(d, "ml_predictions.parquet")) + glob.glob(os.path.join(d, "ml_predictions*.csv")))
        if not paths:
            self.log(f"  no ml_predictions.parquet / .csv in {d}")
            return
        p = paths[0]
        df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        df["date_time"] = pd.to_datetime(df["date_time"])
        df = df.drop_duplicates("date_time").set_index("date_time")
        for split in ("val", "test"):
            part = df.reindex(pd.to_datetime(self.wd.times[split]))
            for c in df.columns:
                if c in ML_META_COLS or not np.issubdtype(df[c].dtype, np.number) or part[c].isna().any():
                    continue
                self.preds[split][ML_PREFIX + c] = part[c].to_numpy(float)
        self.ml_names = [n for n in self.preds["test"] if n.startswith(ML_PREFIX)]
        val_ml = {n: float(np.mean(np.abs(self.preds["val"][n] - self.y["val"])))
                  for n in self.ml_names if n in self.preds["val"]}
        if val_ml:
            self.best_ml = min(val_ml, key=val_ml.get)
        self.log(f"  ML predictions from {os.path.basename(p)}: {len(self.ml_names)} models; "
                 f"best ML on validation: {self.best_ml}")

    def run_dm(self):
        best, y, pt = self.best["model"], self.y["test"], self.preds["test"]
        others = ([self.best_ml] if self.best_ml else []) + [n for n in pt if n not in (best, self.best_ml)]
        rows = []
        for o in others:
            r = diebold_mariano(y, pt[best], pt[o], lags=24)
            rows.append(dict(model_a=best, model_b=o, primary=o == self.best_ml, **r,
                             better=("A" if r["mean_d"] < 0 else "B") if r["p_value"] < 0.05 else "no significant difference"))
        self.dm = pd.DataFrame(rows)

    # ------------------------------------------------------------------ tables / run
    def results_table(self) -> pd.DataFrame:
        rows = []
        for split in ("val", "test"):
            for name, p in self.preds[split].items():
                family = "ML" if name.startswith(ML_PREFIX) else "Baseline" if name in BASELINES else "DL"
                variant = "" if family != "DL" else ("ensemble, train only" if name.endswith(BASE_SUFFIX)
                                                     else "ensemble, fine-tuned on train + val")
                rows.append(dict(model=name, family=family, variant=variant, split=split,
                                 **all_metrics(self.y[split], p, self.scale)))
        return pd.DataFrame(rows)

    def run_all(self):
        steps = [("data", self.load_data), ("baselines", self.run_baselines), ("tuning", self.run_tuning),
                 ("final multi-seed training + fine-tuning", self.run_final), ("selection", self.select_best),
                 ("ablation", self.run_ablation), ("explanation", self.run_explain),
                 ("ML predictions", self.load_ml_predictions), ("Diebold-Mariano tests", self.run_dm)]
        for name, fn in steps:
            self.log(f"[{name}]")
            fn()
        self.save()
        t = self.results_table()
        return t[t["split"] == "test"].set_index("model").drop(columns="split").sort_values("MAE")

    def save(self):
        from .artifacts import save_experiment
        save_experiment(self)
        self.log(f"artifacts written to {os.path.abspath(self.cfg.out_dir)}")
