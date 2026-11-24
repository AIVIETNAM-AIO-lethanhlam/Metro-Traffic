"""Result tables and static figures built from the saved artifacts.

Used by `export_results.py` and the dashboard's export button. Needs only numpy, pandas and
matplotlib (no PyTorch).

    from metro_traffic.artifacts import load_artifacts
    from metro_traffic.report import export_results
    export_results(load_artifacts("outputs"), "result")
"""
from __future__ import annotations

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .baselines import BASELINES  # noqa: E402

BASE_SUFFIX = " (no fine-tune)"
REFERENCE = "Hour-of-week mean"            # strongest naive reference
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_COLORS = {"Weekday": "#1f77b4", "Weekend": "#2ca02c", "Holiday": "#d62728"}
FAMILY_COLORS = {"Baseline": "#e07b39", "DL": "#1f4e9c", "DL (no fine-tune)": "#8fb0de", "ML": "#2e8b57"}
BASELINE_COLORS = {"Persistence (t-1)": "#8c564b", "Seasonal naive (t-24)": "#d62728",
                   "Hour-of-week mean": "#e0a100"}
DL_COLORS = {"LSTM": "#1f4e9c", "TCN": "#7b3fa0"}
BLUE_SCALE = ["#eef3fa", "#cddcf0", "#9dbbe2", "#6896cf", "#3a6db3", "#123f7d"]
INDIGO_SCALE = ["#f1f1f8", "#d7d5ec", "#b2acd9", "#8a7fc2", "#6151a6", "#382a7a"]
SPLIT_COLORS = {"train": "#1f4e9c", "val": "#e07b39", "test": "#2ca02c"}


def plotly_scale(colors):
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


def mpl_cmap(colors, name="custom"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def density_dots(x, y, lim, bins: int = 90):
    """One dot per non-empty cell of a 2-D histogram (centres and counts), densest drawn last."""
    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=[[0, lim], [0, lim]])
    ix, iy = np.nonzero(hist)
    cx, cy = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    cnt = hist[ix, iy]
    order = np.argsort(cnt)
    return cx[ix][order], cy[iy][order], cnt[order]


# ----------------------------------------------------------------- helpers
def best_model_name(A) -> str:
    if A.get("best"):
        return A["best"]["model"]
    r = A["results"]
    dl = r[(r["family"] == "DL") & (r["split"] == "val")]
    return dl.sort_values("MAE")["model"].iloc[0].replace(BASE_SUFFIX, "") if len(dl) else REFERENCE


def family(name: str) -> str:
    if name in BASELINES:
        return "Baseline"
    if name.startswith("ML: "):
        return "ML"
    return "DL (no fine-tune)" if name.endswith(BASE_SUFFIX) else "DL"


def color_of(name: str) -> str:
    if name in BASELINE_COLORS:
        return BASELINE_COLORS[name]
    kind = name.replace(BASE_SUFFIX, "")
    if kind in DL_COLORS:
        return DL_COLORS[kind]
    return "#2e8b57" if name.startswith("ML: ") else "#444444"


def leaderboard(A, split: str = "test") -> pd.DataFrame:
    r = A["results"]
    r = r[r["split"] == split].drop(columns=["split"]).set_index("model")
    return r.sort_values("MAE")


def final_models(A) -> list[str]:
    """Fine-tuned DL ensembles + baselines + ML, i.e. the rows compared on test."""
    return [m for m in A["predictions"]["test"]["preds"] if not m.endswith(BASE_SUFFIX)]


def comparison_models(A) -> list[str]:
    best = best_model_name(A)
    names = A["predictions"]["test"]["preds"]
    ml = best_ml_name(A)
    other_dl = [m for m in DL_COLORS if m != best]
    return list(dict.fromkeys([m for m in [best, ml, *other_dl, REFERENCE] if m and m in names]))


def best_ml_name(A):
    """ML model with the lowest validation MAE (group selection rule), if ML predictions were loaded."""
    r = A["results"]
    ml = r[(r["family"] == "ML") & (r["split"] == "val")]
    return ml.sort_values("MAE")["model"].iloc[0] if len(ml) else None


def error_frame(A, split: str = "test") -> pd.DataFrame:
    """One row per target hour with absolute errors of every model."""
    P = A["predictions"][split]
    t = pd.DatetimeIndex(P["times"])
    df = pd.DataFrame({"time": t, "y": P["y"], "hour": t.hour, "dow": t.dayofweek, "month": t.month,
                       "day_type": P["day_type"], "weather": P["weather"]})
    for m, p in P["preds"].items():
        df[m] = np.abs(p - P["y"])
    return df


def mae_by(A, key: str, models=None, split: str = "test") -> pd.DataFrame:
    df = error_frame(A, split)
    models = models or final_models(A)
    out = df.groupby(key)[models].mean()
    if key == "dow":
        out.index = [WEEKDAYS[i] for i in out.index]
    out["n"] = df.groupby(key).size()
    return out


def seed_summary(A) -> pd.DataFrame:
    """Per encoder: mean +- std of the single-seed MAEs and the MAE of the seed ensemble."""
    s = A.get("seeds")
    if s is None:
        return pd.DataFrame()
    test = leaderboard(A, "test")["MAE"]
    val = leaderboard(A, "val")["MAE"]
    g = s.groupby("model")
    out = pd.DataFrame({
        "seeds": g.size(),
        "val_MAE_mean": g["val_MAE"].mean(), "val_MAE_std": g["val_MAE"].std(ddof=0),
        "val_MAE_ensemble": [val.get(m + BASE_SUFFIX, np.nan) for m in g.groups],
        "test_MAE_mean": g["test_MAE"].mean(), "test_MAE_std": g["test_MAE"].std(ddof=0),
        "test_MAE_ensemble": [test.get(m, np.nan) for m in g.groups],
        "test_MAE_ensemble_no_finetune": [test.get(m + BASE_SUFFIX, np.nan) for m in g.groups],
        "n_params": g["n_params"].first(),
        "epochs_mean": g["epochs"].mean(), "seconds_total": g["seconds"].sum()})
    return out.sort_values("val_MAE_ensemble")


def ablation_summary(A) -> pd.DataFrame:
    ab = A.get("ablation")
    if ab is None:
        return pd.DataFrame()
    g = ab.groupby(["stage", "features"], sort=True)
    out = g.agg(n_numeric=("n_numeric", "first"), weather_main=("weather_main", "first"), seeds=("seed", "size"),
                val_MAE=("val_MAE", "mean"), val_std=("val_MAE", lambda s: s.std(ddof=0)),
                test_MAE=("test_MAE", "mean"), test_std=("test_MAE", lambda s: s.std(ddof=0))).reset_index()
    out["val_gain_vs_previous"] = -out["val_MAE"].diff()
    return out


def best_window_start(times, days: int = 14) -> pd.Timestamp:
    """Monday at 00:00 whose following `days` days hold the most test hours (avoids data gaps)."""
    t = pd.DatetimeIndex(times)
    first = t[0].normalize() + pd.Timedelta(days=(7 - t[0].dayofweek) % 7)
    starts = pd.date_range(first, t[-1] - pd.Timedelta(days=days), freq="7D")
    if len(starts) == 0:
        return t[0].normalize()
    counts = [((t >= s) & (t < s + pd.Timedelta(days=days))).sum() for s in starts]
    return starts[int(np.argmax(counts))]


def worst_hours(A, model: str, n: int = 15) -> pd.DataFrame:
    P = A["predictions"]["test"]
    e = P["preds"][model] - P["y"]
    idx = np.argsort(-np.abs(e))[:n]
    return pd.DataFrame({"time": pd.DatetimeIndex(P["times"])[idx], "actual": P["y"][idx],
                         "forecast": P["preds"][model][idx], "error": e[idx],
                         "day_type": P["day_type"][idx], "weather": P["weather"][idx]})


def worst_days(A, model: str, n: int = 10) -> pd.DataFrame:
    df = error_frame(A)
    df["date"] = df["time"].dt.date
    g = df.groupby("date").agg(MAE=(model, "mean"), hours=("y", "size"), day_type=("day_type", "first"))
    return g[g["hours"] >= 12].sort_values("MAE", ascending=False).head(n)


# ------------------------------------------------------------------ figures
def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.3, "legend.frameon": False, "figure.facecolor": "white"})


def _save(fig, path, files):
    fig.savefig(path)
    plt.close(fig)
    files.append(path)


def export_results(A, out_dir: str = "result") -> list[str]:
    """Write figures (PNG), tables (CSV) and a markdown summary into `out_dir`."""
    _style()
    fig_dir, tab_dir = os.path.join(out_dir, "figures"), os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    files: list[str] = []
    F = lambda name: os.path.join(fig_dir, name)  # noqa: E731
    T = lambda name: os.path.join(tab_dir, name)  # noqa: E731

    def table(df, name, index=True, digits=3):
        if df is not None and len(df):
            df.round({c: digits for c in df.select_dtypes('number').columns}).to_csv(T(name), index=index)
            files.append(T(name))

    P = A["predictions"]
    best = best_model_name(A)
    cmp = comparison_models(A)
    lb_test, lb_val = leaderboard(A, "test"), leaderboard(A, "val")
    seeds = seed_summary(A)
    abl = ablation_summary(A)

    # ---------------- tables
    table(lb_test, "leaderboard_test.csv")
    table(lb_val, "leaderboard_val.csv")
    table(seeds, "seed_summary.csv")
    table(A.get("seeds"), "seed_runs.csv", index=False)
    table(abl, "ablation_summary.csv", index=False)
    for key in ["hour", "dow", "day_type", "month", "weather"]:
        table(mae_by(A, key), f"mae_by_{key}.csv", digits=2)
    table(worst_days(A, best), "worst_days_best_model.csv", digits=1)
    table(worst_hours(A, best), "worst_hours_best_model.csv", index=False, digits=1)
    for key, name in [("screening", "screening.csv"), ("tuning", "tuning_log.csv"), ("dm", "diebold_mariano.csv"),
                      ("ig_features", "ig_feature_importance.csv"), ("occlusion", "weather_occlusion.csv"),
                      ("splits", "data_splits.csv")]:
        table(A.get(key), name, index=False, digits=4)

    # ---------------- 1. leaderboard (test)
    r = lb_test.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.42 * len(r) + 1.8))
    cols = [FAMILY_COLORS[family(m)] for m in r.index]
    for a, metric, title in [(ax[0], "MAE", "Test MAE (vehicles/hour, lower is better)"),
                             (ax[1], "MASE", "Test MASE (MAE / in-sample seasonal-naive MAE)")]:
        bars = a.barh(r.index, r[metric], color=cols)
        a.bar_label(bars, fmt="%.1f" if metric == "MAE" else "%.3f", padding=3, fontsize=9)
        a.set_title(title)
        a.grid(axis="y", visible=False)
    ax[1].axvline(1, color="k", ls="--", lw=1)
    ax[1].set_yticklabels([])
    fams = [f for f in FAMILY_COLORS if any(family(m) == f for m in r.index)]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in fams], fams, loc="lower center",
               ncol=len(fams), bbox_to_anchor=(0.5, -0.04))
    _save(fig, F("01_leaderboard_test.png"), files)

    # ---------------- 2. seeds: single runs vs ensemble
    s = A.get("seeds")
    if s is not None and len(s):
        order = seeds.index.tolist()
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
        for a, col, ens, title in [(ax[0], "val_MAE", "val_MAE_ensemble", "Validation MAE (selection metric)"),
                                   (ax[1], "test_MAE", "test_MAE_ensemble", "Test MAE after fine-tuning")]:
            for i, m in enumerate(order):
                v = s.loc[s["model"] == m, col].to_numpy()
                a.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), color=color_of(m), s=40,
                          alpha=0.75, label="single seed" if i == 0 else None)
                a.scatter(seeds.loc[m, ens], i, marker="D", s=90, color="k", label="seed ensemble" if i == 0 else None)
            a.set_yticks(range(len(order)), order)
            a.set_title(title)
            a.set_xlabel("MAE (vehicles/hour)")
            a.legend(loc="lower right")
        fig.suptitle(f"Every encoder trained with seeds {', '.join(str(x) for x in A['config']['seeds'])}", y=1.02)
        _save(fig, F("02_seeds_and_ensembles.png"), files)

    # ---------------- 3. fine-tuning effect
    if len(seeds):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(seeds))
        ax.bar(x - 0.2, seeds["test_MAE_ensemble_no_finetune"], 0.4, color=FAMILY_COLORS["DL (no fine-tune)"],
               label="trained on train only")
        ax.bar(x + 0.2, seeds["test_MAE_ensemble"], 0.4, color=FAMILY_COLORS["DL"], label="fine-tuned on train + val")
        ax.set_xticks(x, seeds.index)
        ax.set_ylabel("Test MAE")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f", padding=2, fontsize=9)
        lo = float(np.nanmin(seeds[["test_MAE_ensemble", "test_MAE_ensemble_no_finetune"]].to_numpy()))
        hi = float(np.nanmax(seeds[["test_MAE_ensemble", "test_MAE_ensemble_no_finetune"]].to_numpy()))
        ax.set_ylim(max(0.0, lo - 0.25 * (hi - lo + 10)), hi + 0.25 * (hi - lo + 10))
        ax.set_title("Effect of fine-tuning on train + validation (seed ensembles)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        _save(fig, F("03_finetune_effect.png"), files)

    # ---------------- 4. one-step forecasts over two test weeks
    t = pd.DatetimeIndex(P["test"]["times"])
    start = best_window_start(t, 14)
    full = pd.date_range(start, start + pd.Timedelta(days=14), freq="h", inclusive="left")
    pos = pd.Series(np.arange(len(t)), index=t).reindex(full)       # NaN where an hour is missing -> gap
    ok = pos.notna().to_numpy()
    ii = pos.fillna(0).astype(int).to_numpy()

    def seg(v):
        return np.where(ok, np.asarray(v, float)[ii], np.nan)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(full, seg(P["test"]["y"]), color="k", lw=2, label="Actual")
    for name in cmp:
        ax.plot(full, seg(P["test"]["preds"][name]), color=color_of(name), lw=2 if name == best else 1.4,
                ls="-" if name == best else "--", label=name)
    ax.set_title(f"One-hour-ahead forecasts, two test weeks from {start:%Y-%m-%d}")
    ax.set_ylabel("vehicles/hour")
    ax.legend(ncol=len(cmp) + 1, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _save(fig, F("04_forecasts_two_weeks.png"), files)

    # ---------------- 5. error breakdown
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for a, key, title in [(ax[0, 0], "hour", "MAE by hour of day"), (ax[0, 1], "dow", "MAE by day of week"),
                          (ax[1, 0], "day_type", "MAE by day type"), (ax[1, 1], "month", "MAE by month")]:
        d = mae_by(A, key, cmp)
        if key in ("hour",):
            for name in cmp:
                a.plot(d.index, d[name], color=color_of(name), marker="o", ms=3, lw=2 if name == best else 1.3,
                       label=name)
        else:
            w = 0.8 / len(cmp)
            x = np.arange(len(d))
            for i, name in enumerate(cmp):
                a.bar(x + (i - (len(cmp) - 1) / 2) * w, d[name], w, color=color_of(name), label=name)
            a.set_xticks(x, [f"{i}\n(n={n})" for i, n in zip(d.index, d["n"])] if key == "day_type" else d.index)
        a.set_title(title)
        a.set_ylabel("MAE")
    ax[0, 0].legend()
    _save(fig, F("05_error_breakdown.png"), files)

    # ---------------- 6. predicted vs actual + residuals by hour
    y, p = P["test"]["y"], P["test"]["preds"][best]
    lim = float(max(y.max(), p.max()) * 1.03)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    cx, cy, cnt = density_dots(y, p, lim)
    sc = ax[0].scatter(cx, cy, c=cnt, cmap=mpl_cmap(BLUE_SCALE[1:]), s=14, norm=matplotlib.colors.LogNorm())
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="Actual", ylabel="Forecast", title=f"{best}: forecast vs actual")
    fig.colorbar(sc, ax=ax[0], label="hours per cell")
    res = pd.DataFrame({"hour": t.hour, "res": p - y})
    ax[1].boxplot([res.loc[res.hour == h, "res"] for h in range(24)], positions=range(24), showfliers=False)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel="hour of day", ylabel="forecast - actual", title="Residuals by hour (test)")
    _save(fig, F("06_residuals.png"), files)

    # ---------------- 7. ablation
    if len(abl):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = np.arange(len(abl))
        ax.errorbar(x - 0.08, abl["val_MAE"], yerr=abl["val_std"], marker="o", lw=2, capsize=4, label="validation")
        ax.errorbar(x + 0.08, abl["test_MAE"], yerr=abl["test_std"], marker="s", lw=2, capsize=4, label="test")
        ab = A["ablation"]
        for i, st in enumerate(abl["stage"]):
            v = ab.loc[ab["stage"] == st, "val_MAE"]
            ax.scatter(np.full(len(v), i - 0.08), v, color="#1f4e9c", s=12, alpha=0.5)
        ax.set_xticks(x, [f"{f}\n({n} numeric{' + wx' if w else ''})" for f, n, w in
                          zip(abl["features"], abl["n_numeric"], abl["weather_main"])])
        ax.set_ylabel("MAE (mean ± std over seeds)")
        ax.set_title(f"Cumulative feature-group ablation ({best})")
        ax.legend()
        _save(fig, F("07_ablation.png"), files)

    # ---------------- 8. explanation
    igf, igl = A.get("ig_features"), A.get("ig_lag")
    if igf is not None and igl is not None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
        top = igf.head(15).iloc[::-1]
        ax[0].barh(top["feature"], top["mean_abs_attr"], color="#1f4e9c")
        ax[0].set_title("Integrated Gradients: top 15 features")
        ax[0].set_xlabel("mean |IG| (standardised units)")
        ax[1].plot(igl["lag"], igl["mean_abs_ig"], color="#1f4e9c")
        for h in (24, 48, 72, 96, 120, 144):
            ax[1].axvline(-h, color="#999", lw=0.7, ls=":")
        ax[1].set(xlabel="lag relative to the target hour (0 = row of hour t)", ylabel="mean |IG|",
                  title="Attribution over the 168-hour input window")
        _save(fig, F("08_integrated_gradients.png"), files)
    occ = A.get("occlusion")
    if occ is not None and len(occ):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        o = occ.sort_values("mean_abs_delta")
        ax.barh(o["weather"], o["mean_abs_delta"], color="#7b3fa0")
        ax.set(title="Weather occlusion: category replaced by the unknown token",
               xlabel="mean |change of the prediction| (vehicles/hour)")
        _save(fig, F("09_weather_occlusion.png"), files)

    # ---------------- 10. learning curves of the best encoder
    H = A.get("histories") or {}
    curves = {k: v for k, v in H.items() if k.startswith(best + " | seed") and "fine-tune" not in k}
    if curves:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
        for k, h in curves.items():
            ax[0].plot(h["epoch"], h["val_mae"], label=k.split("| ")[1])
            ax[1].plot(h["epoch"], h["lr"], label=k.split("| ")[1])
        ax[0].set(title=f"{best}: validation MAE per epoch", xlabel="epoch", ylabel="val MAE")
        ax[1].set(title="Learning rate (ReduceLROnPlateau)", xlabel="epoch", yscale="log")
        ax[0].legend()
        _save(fig, F("10_learning_curves.png"), files)

    path = write_summary(A, out_dir, files)
    files.append(path)
    return files


def write_summary(A, out_dir: str, files: list[str]) -> str:
    best = best_model_name(A)
    lb = leaderboard(A, "test")
    b = A.get("best") or {}
    seeds = seed_summary(A)
    abl = ablation_summary(A)
    ref = lb.loc[REFERENCE, "MAE"] if REFERENCE in lb.index else np.nan
    bm = lb.loc[best]
    L = [f"# Results summary - Metro traffic, one-hour-ahead DL ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"* Selected encoder (lowest validation MAE of the base seed ensemble): **{best}**"
         + (" (Optuna-tuned)" if b.get("tuned") else " (default hyper-parameters)"),
         f"* Seeds per ensemble: {', '.join(str(s) for s in A['config']['seeds'])}; "
         f"fine-tuned on train + val for {A['config']['finetune_epochs']} epochs",
         f"* Test: MAE {bm['MAE']:.1f}, RMSE {bm['RMSE']:.1f}, R2 {bm['R2']:.3f}, sMAPE {bm['sMAPE']:.2f} %, "
         f"MASE {bm.get('MASE', np.nan):.3f}",
         f"* Strongest naive reference ({REFERENCE}): test MAE {ref:.1f} "
         f"-> {100 * (1 - bm['MAE'] / ref):.1f} % lower MAE" if np.isfinite(ref) else "", "",
         "## Test leaderboard", "", lb.round(3).to_markdown() if _has_tabulate() else lb.round(3).to_string(), ""]
    if len(seeds):
        L += ["## Seeds (single runs vs ensemble)", "",
              seeds.round(2).to_markdown() if _has_tabulate() else seeds.round(2).to_string(), ""]
    if len(abl):
        L += ["## Ablation (mean over seeds)", "",
              abl.round(2).to_markdown(index=False) if _has_tabulate() else abl.round(2).to_string(index=False), ""]
    dm = A.get("dm")
    if dm is not None and len(dm):
        L += ["## Diebold-Mariano tests (absolute errors, Newey-West 24 lags)", "",
              dm.round(4).to_markdown(index=False) if _has_tabulate() else dm.round(4).to_string(index=False), ""]
    L += ["## Files", ""] + [f"* `{os.path.relpath(f, out_dir)}`" for f in files]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
