"""Result tables and static figures built from the saved artifacts.

Used by `export_results.py` and the dashboard's "Export results" button. Needs only
numpy, pandas and matplotlib (no PyTorch, no scikit-learn).

    from metro_traffic.artifacts import load_artifacts
    from metro_traffic.report import export_results
    export_results(load_artifacts("outputs"), "result")
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASELINES = ("Daily-naive", "Seasonal-naive (weekly)", "Ridge (linear, lags)")
SNAIVE, RIDGE = "Seasonal-naive (weekly)", "Ridge (linear, lags)"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_COLORS = {"Weekday": "#1f77b4", "Weekend": "#2ca02c", "Holiday": "#d62728"}
FIXED_COLORS = {SNAIVE: "#d62728", RIDGE: "#ff7f0e", "Daily-naive": "#8c564b"}
# soft sequential scales that stay visible on white (no yellow): light grey-blue -> deep blue / indigo
BLUE_SCALE = ["#eef3fa", "#cddcf0", "#9dbbe2", "#6896cf", "#3a6db3", "#123f7d"]
INDIGO_SCALE = ["#f1f1f8", "#d7d5ec", "#b2acd9", "#8a7fc2", "#6151a6", "#382a7a"]


def plotly_scale(colors):
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


def mpl_cmap(colors, name="custom"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def density_dots(x, y, lim, bins: int = 90):
    """Bin (x, y) pairs on a grid and return one dot per non-empty cell: centres and counts.
    Plotted as round markers coloured by count, this reads much cleaner than square heatmap cells."""
    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=[[0, lim], [0, lim]])
    ix, iy = np.nonzero(hist)
    cx, cy = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    cnt = hist[ix, iy]
    order = np.argsort(cnt)                       # dense dots drawn last (on top)
    return cx[ix][order], cy[iy][order], cnt[order]


DL_COLORS = ["#1f4e9c", "#2ca02c", "#9467bd", "#17becf", "#e377c2", "#bcbd22", "#7f7f7f", "#393b79"]


# ----------------------------------------------------------------- helpers
def best_model_name(A) -> str:
    if A.get("best"):
        return A["best"]["name"]
    dl = [m for m in A["results"].index if m.startswith("DL-")]
    return dl[0] if dl else A["results"].index[0]


def model_kind(name: str) -> str:
    if name in BASELINES:
        return "Baseline"
    if name.startswith("Ablation"):
        return "Ablation"
    return "Deep learning"


def color_map(A) -> dict:
    """Stable, high-contrast colours: best model dark blue, baselines warm colours."""
    best = best_model_name(A)
    cmap, i = {"Actual": "#111111", best: DL_COLORS[0]}, 1
    for m in A["predictions"]["preds"]:
        if m in cmap:
            continue
        if m in FIXED_COLORS:
            cmap[m] = FIXED_COLORS[m]
        elif m.startswith("Ablation"):
            cmap[m] = "#7f7f7f"
        else:
            cmap[m] = DL_COLORS[i % len(DL_COLORS)]
            i += 1
    return cmap


def target_index(A) -> pd.DataFrame:
    """One row per (test window, horizon) target, in the same order as pred.ravel()."""
    P = A["predictions"]
    t = pd.DatetimeIndex(P["target_times"].ravel())
    return pd.DataFrame({"time": t, "y": P["y_true"].ravel(), "day_type": P["day_type"].ravel(),
                         "hour": t.hour, "month": t.month,
                         "horizon": np.tile(np.arange(1, P["y_true"].shape[1] + 1), P["y_true"].shape[0])})


def mae_by_horizon(A, models=None) -> pd.DataFrame:
    P = A["predictions"]
    models = models or list(P["preds"])
    return pd.DataFrame({m: np.abs(P["preds"][m] - P["y_true"]).mean(0) for m in models},
                        index=pd.Index(np.arange(1, P["y_true"].shape[1] + 1), name="horizon"))


def mae_by(A, key: str, models=None) -> pd.DataFrame:
    """MAE grouped by 'hour', 'day_type' or 'month'."""
    P = A["predictions"]
    base = target_index(A)
    models = models or list(P["preds"])
    out = {m: np.abs(P["preds"][m].ravel() - base.y).groupby(base[key]).mean() for m in models}
    df = pd.DataFrame(out)
    if key == "day_type":
        df = df.reindex([d for d in ["Weekday", "Weekend", "Holiday"] if d in df.index])
    return df


def worst_days(A, model: str, n: int = 10) -> pd.DataFrame:
    base = target_index(A)
    base["pred"] = A["predictions"]["preds"][model].ravel()
    base["abs_err"] = (base.pred - base.y).abs()
    return (base.assign(date=base.time.dt.date).groupby("date")
            .agg(MAE=("abs_err", "mean"), mean_actual=("y", "mean"), mean_pred=("pred", "mean"),
                 day_type=("day_type", "first"))
            .sort_values("MAE", ascending=False).head(n))


def stitched_days(A, start, days: int = 7):
    """Day-ahead forecasts issued at 23:00, stitched over `days` days. Returns (times, row indices)."""
    P = A["predictions"]
    tt = P["target_times"]
    origin = pd.DatetimeIndex(tt[:, 0]) - pd.Timedelta(hours=1)
    pos = {t: i for i, t in enumerate(origin)}
    idx = [pos[t] for k in range(days)
           if (t := pd.Timestamp(start) + pd.Timedelta(days=k) - pd.Timedelta(hours=1)) in pos]
    return pd.DatetimeIndex(tt[idx].ravel()), idx


def comparison_models(A) -> list:
    best = best_model_name(A)
    names = A["predictions"]["preds"]
    return list(dict.fromkeys([m for m in [best, RIDGE, SNAIVE] if m in names]))


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

    RES, P = A["results"], A["predictions"]
    best = best_model_name(A)
    C = color_map(A)
    cmp = comparison_models(A)

    # ---------------- tables
    RES.to_csv(T("leaderboard_test.csv")); files.append(T("leaderboard_test.csv"))
    hz = mae_by_horizon(A); hz.round(2).to_csv(T("mae_by_horizon.csv")); files.append(T("mae_by_horizon.csv"))
    for key in ["hour", "day_type", "month"]:
        mae_by(A, key).round(2).to_csv(T(f"mae_by_{key}.csv")); files.append(T(f"mae_by_{key}.csv"))
    wd = worst_days(A, best); wd.round(1).to_csv(T("worst_days_best_model.csv")); files.append(T("worst_days_best_model.csv"))
    for key, name in [("perm", "permutation_importance.csv"), ("ig_features", "ig_feature_importance.csv"),
                      ("tuning", "tuning_log.csv"), ("splits", "data_splits.csv")]:
        if A.get(key) is not None:
            A[key].to_csv(T(name), index=False); files.append(T(name))

    # ---------------- 1. leaderboard
    r = RES.sort_values("MAE", ascending=False)
    kind_col = {"Baseline": "#e07b39", "Deep learning": "#1f4e9c", "Ablation": "#9e9e9e"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.45 * len(r) + 1.8))
    cols = [kind_col[model_kind(m)] for m in r.index]
    for a, metric, title in [(ax[0], "MAE", "Test MAE (vehicles/hour, lower is better)"),
                             (ax[1], "rMAE", "rMAE = MAE / seasonal-naive MAE")]:
        bars = a.barh(r.index, r[metric], color=cols)
        a.bar_label(bars, fmt="%.1f" if metric == "MAE" else "%.3f", padding=3, fontsize=9)
        a.set_title(title); a.grid(axis="y", visible=False)
    ax[1].axvline(1, color="k", ls="--", lw=1); ax[1].set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in kind_col.values()]
    fig.legend(handles, kind_col.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    _save(fig, F("01_leaderboard.png"), files)

    # ---------------- 2. horizon
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in hz.columns:
        if m == "Daily-naive" or m.startswith("Ablation"):
            continue
        ax.plot(hz.index, hz[m], marker="o", ms=3, lw=2.6 if m == best else 1.6, color=C[m],
                ls="--" if m in BASELINES else "-", label=m)
    ax.set(xlabel="Hours ahead", ylabel="MAE (vehicles/hour)", title="Error by forecast horizon")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, F("02_mae_by_horizon.png"), files)

    # ---------------- 3-4. data patterns
    hd = A.get("hourly")
    if hd is not None:
        hd = hd[hd.valid == 1].copy()
        hd["hour"], hd["weekday"] = hd.date_time.dt.hour, hd.date_time.dt.dayofweek
        piv = hd.pivot_table(index="weekday", columns="hour", values="traffic_volume", aggfunc="mean") / 1000
        fig, ax = plt.subplots(figsize=(14, 4.2))
        im = ax.imshow(piv.values, cmap=mpl_cmap(BLUE_SCALE), aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > piv.values.max() * 0.6 else "#222")
        ax.set_xticks(range(24)); ax.set_yticks(range(7), WEEKDAYS); ax.grid(False)
        ax.set(xlabel="Hour of day", title="Average traffic (thousand vehicles/hour): weekday × hour")
        fig.colorbar(im, ax=ax, pad=0.01, label="k veh/h")
        _save(fig, F("03_weekday_hour_heatmap.png"), files)

        hd["day_type"] = np.select([hd.is_holiday == 1, hd.is_weekend == 1], ["Holiday", "Weekend"], "Weekday")
        prof = hd.groupby(["day_type", "hour"]).traffic_volume.mean().unstack(0)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for d in ["Weekday", "Weekend", "Holiday"]:
            if d in prof:
                ax.plot(prof.index, prof[d], lw=2.4, marker="o", ms=3, color=DAY_COLORS[d], label=d)
        ax.set(xlabel="Hour of day", ylabel="vehicles/hour", title="Hourly profile by day type", xticks=range(0, 24, 2))
        ax.legend()
        _save(fig, F("04_day_type_profile.png"), files)

    # ---------------- 5. forecast examples
    weeks = [("2018-03-05", "regular week"), ("2018-05-26", "Memorial Day week"), ("2018-07-01", "Independence Day week")]
    fig, axes = plt.subplots(len(weeks), 1, figsize=(15, 4 * len(weeks)))
    for ax, (start, label) in zip(axes, weeks):
        x, idx = stitched_days(A, start)
        if not idx:
            ax.set_visible(False); continue
        ax.plot(x, P["y_true"][idx].ravel(), color="black", lw=2.2, label="Actual")
        for m in cmp:
            if m == RIDGE:
                continue
            ax.plot(x, P["preds"][m][idx].ravel(), color=C[m], lw=1.8 if m == best else 1.3,
                    ls="--" if m in BASELINES else "-", label=m)
        ax.set_title(f"Day-ahead forecasts, {label} (starting {start})"); ax.set_ylabel("vehicles/hour")
        ax.legend(loc="upper right", ncol=3, fontsize=9)
    fig.tight_layout()
    _save(fig, F("05_forecast_examples.png"), files)

    # ---------------- 6. error breakdown
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    g = mae_by(A, "hour", cmp)
    for m in cmp:
        ax[0].plot(g.index, g[m], lw=2.4 if m == best else 1.6, marker="o", ms=3, color=C[m], label=m)
    ax[0].set(title="MAE by hour of day", xlabel="Hour of day", ylabel="MAE", xticks=range(0, 24, 3))
    g = mae_by(A, "day_type", cmp)
    w = 0.8 / len(cmp)
    for k, m in enumerate(cmp):
        b = ax[1].bar(np.arange(len(g)) + k * w - 0.4 + w / 2, g[m], width=w, color=C[m], label=m)
        ax[1].bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax[1].set_xticks(range(len(g)), g.index); ax[1].set_title("MAE by day type"); ax[1].grid(axis="x", visible=False)
    g = mae_by(A, "month", cmp)
    for m in cmp:
        ax[2].plot(g.index, g[m], lw=2.4 if m == best else 1.6, marker="o", color=C[m], label=m)
    ax[2].set(title="MAE by month (2018)", xlabel="Month")
    ax[0].legend(fontsize=9)
    _save(fig, F("06_error_breakdown.png"), files)

    # ---------------- 7. predicted vs actual (own figure) + residuals
    y, p = P["y_true"].ravel(), P["preds"][best].ravel()
    lim = float(max(y.max(), p.max()))
    from matplotlib.colors import LogNorm
    dx, dy, dc = density_dots(y, p, lim, bins=160)
    fig, ax = plt.subplots(figsize=(15, 7))
    sc = ax.scatter(dx, dy, c=dc, s=7, cmap=mpl_cmap(BLUE_SCALE[1:]), norm=LogNorm(vmin=1, vmax=dc.max()),
                    edgecolors="none")
    ax.plot([0, lim], [0, lim], color="#d62728", lw=1.3, ls="--", label="perfect forecast")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xticks(np.arange(0, lim + 1, 500))
    ax.set(xlabel="Actual (veh/h)", ylabel="Predicted (veh/h)", title=f"Predicted vs actual — {best}")
    ax.legend(loc="upper left")
    fig.colorbar(sc, ax=ax, label="number of hours (log scale)", pad=0.015)
    _save(fig, F("07_pred_vs_actual.png"), files)

    res = p - y
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(res, bins=np.arange(-2500, 2501, 50), color="#1f4e9c", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#d62728", lw=1.3)
    ax.set(xlabel="Prediction − actual (veh/h)", ylabel="count",
           title=f"Residuals of {best} (mean {res.mean():.1f}, std {res.std():.0f})")
    _save(fig, F("07b_residuals.png"), files)

    # ---------------- 8. explanations
    if A.get("perm") is not None:
        pm = A["perm"].sort_values("delta_MAE")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        b = ax.barh(pm.group, pm.delta_MAE, xerr=pm["std"], color="#1f4e9c")
        ax.bar_label(b, fmt="%.0f", padding=4, fontsize=9)
        ax.set(xlabel="Increase in test MAE when the group is shuffled (veh/h)",
               title=f"Permutation importance — {best} (base MAE {pm.base_MAE.iloc[0]:.0f})")
        ax.grid(axis="y", visible=False)
        _save(fig, F("08_permutation_importance.png"), files)
    if A.get("ig_lag") is not None:
        lag = A["ig_lag"].sort_values("hours_before_origin")
        smooth = lag.importance.rolling(3, center=True, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(lag.hours_before_origin, lag.importance, color="#9ecae1", lw=1, label="raw")
        ax.plot(lag.hours_before_origin, smooth, color="#1f4e9c", lw=2.4, label="3-hour moving average")
        for d in range(24, 169, 24):
            ax.axvline(d, color="#888", ls=":", lw=1)
            ax.text(d, 0.97, f"{d // 24} day" + ("s" if d > 24 else ""), transform=ax.get_xaxis_transform(),
                    ha="right", va="top", rotation=90, fontsize=9, color="#555")
        ax.invert_xaxis()
        ax.set(xlabel="Hours before the forecast origin", ylabel="mean |IG|",
               title="Integrated Gradients by time lag (which past hours the model uses)")
        ax.legend(loc="upper center")
        _save(fig, F("09_ig_by_lag.png"), files)
    if A.get("ig_features") is not None:
        f = A["ig_features"].sort_values("importance")
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(f) + 1.5))
        b = ax.barh(f.feature, f.importance, color="#1f4e9c")
        ax.bar_label(b, fmt="%.2f", padding=3, fontsize=8)
        ax.set(xlabel="sum of mean |IG| over the window", title="Integrated Gradients by input feature")
        ax.grid(axis="y", visible=False)
        _save(fig, F("10_ig_by_feature.png"), files)
    if A.get("ig_heatmap") is not None:
        hm = binned_heatmap(A["ig_heatmap"], bin_hours=3)
        fig, ax = plt.subplots(figsize=(14, 4.2))
        im = ax.imshow(hm.values, cmap=mpl_cmap(INDIGO_SCALE), aspect="auto")
        step = max(1, len(hm.columns) // 14)
        ax.set_xticks(range(0, len(hm.columns), step), hm.columns[::step]); ax.set_yticks(range(len(hm)), hm.index)
        ax.grid(False)
        ax.set(xlabel="Hours before origin (3-hour bins)", ylabel="Forecast step",
               title="Traffic-history attribution per forecast step (row-normalised |IG|)")
        fig.colorbar(im, ax=ax, pad=0.01)
        _save(fig, F("11_ig_horizon_heatmap.png"), files)

    # ---------------- 9. training
    hist = A.get("histories") or {}
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
        for k, h in hist.items():
            ax[0].plot(h.epoch, h.train_loss, lw=1.8, label=k)
            ax[1].plot(h.epoch, h.val_mae, lw=1.8, label=k)
        ax[0].set(title="Training loss (Huber, scaled)", xlabel="epoch")
        ax[1].set(title="Validation MAE (veh/h)", xlabel="epoch")
        ax[1].legend(fontsize=9)
        _save(fig, F("12_learning_curves.png"), files)
    if A.get("tuning") is not None and len(A["tuning"]):
        t = A["tuning"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (m, g) in enumerate(t.groupby("model")):
            ax.scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(g)), g.val_MAE,
                       s=55, alpha=0.85, label=m)
        ax.set_xticks(range(t.model.nunique()), sorted(t.model.unique()))
        ax.set(ylabel="validation MAE", title="Random-search trials")
        _save(fig, F("13_tuning_trials.png"), files)

    files.append(write_summary(A, out_dir, files))
    return files


def binned_heatmap(hm: pd.DataFrame, bin_hours: int = 3, normalise: bool = True) -> pd.DataFrame:
    """Average the IG heatmap over blocks of `bin_hours` lags (less noisy) and normalise each row."""
    cols = np.array([int(c) for c in hm.columns])
    groups = (cols // bin_hours) * bin_hours
    out = hm.T.groupby(groups).mean().T
    out = out[sorted(out.columns, reverse=True)]
    if normalise:
        out = out.div(out.max(axis=1), axis=0)
    return out


def write_summary(A, out_dir: str, files: list[str]) -> str:
    RES = A["results"]
    best = best_model_name(A)
    b = RES.loc[best]
    lines = [f"# Results — I-94 hourly traffic forecasting (24 hours ahead)",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M} from the artifacts in `outputs/`._", "",
             "## Key numbers (test set, Jan–Sep 2018)",
             f"- **Best model (selected on validation): {best}** — MAE {b.MAE:.1f} veh/h, RMSE {b.RMSE:.1f}, "
             f"sMAPE {b.sMAPE:.2f}%, R² {b.R2:.4f}, rMAE {b.rMAE:.3f}"]
    for ref in [SNAIVE, RIDGE]:
        if ref in RES.index:
            lines.append(f"- vs {ref} (MAE {RES.loc[ref, 'MAE']:.1f}): "
                         f"{100 * (1 - b.MAE / RES.loc[ref, 'MAE']):.1f}% lower MAE")
    abl = [m for m in RES.index if m.startswith("Ablation")]
    if abl:
        a = RES.loc[abl[0]]
        diff = a.MAE - b.MAE
        lines.append(f"- Ablation without weather: MAE {a.MAE:.1f} → removing weather "
                     + (f"increases the error by {diff:.1f} veh/h" if diff > 0 else
                        f"*lowers* the error by {-diff:.1f} veh/h (past weather adds noise rather than signal)"))
    if A.get("perm") is not None:
        lines.append("- Most important feature groups (permutation): "
                     + ", ".join(A["perm"].sort_values("delta_MAE", ascending=False).group.head(3)))
    lines += ["", "## Leaderboard", "", "| model | MAE | RMSE | sMAPE (%) | R² | rMAE |", "|---|---|---|---|---|---|"]
    for m, r in RES.iterrows():
        lines.append(f"| {m} | {r.MAE:.1f} | {r.RMSE:.1f} | {r.sMAPE:.2f} | {r.R2:.4f} | {r.rMAE:.3f} |")
    lines += ["", "## Figures", ""]
    for f in files:
        if f.endswith(".png"):
            rel = os.path.relpath(f, out_dir).replace(os.sep, "/")
            title = os.path.splitext(os.path.basename(f))[0].split("_", 1)[1].replace("_", " ").capitalize()
            lines += [f"### {title}", f"![{title}]({rel})", ""]
    lines += ["## Tables", ""] + [f"- `{os.path.relpath(f, out_dir).replace(os.sep, '/')}`"
                                  for f in files if f.endswith(".csv")]
    if A.get("best"):
        lines += ["", "## Selected configuration", "```json", json.dumps(A["best"], indent=2), "```"]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
