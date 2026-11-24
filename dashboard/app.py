"""Streamlit dashboard for the DL part of the Metro Interstate Traffic project
(one-hour-ahead forecasts, several seeds per model).

Run from the project root (so that .streamlit/config.toml is picked up):
    streamlit run dashboard/app.py

It only reads the files that `train.py` or the notebook write to `outputs/`,
so PyTorch and a GPU are not required.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from metro_traffic.artifacts import load_artifacts  # noqa: E402
from metro_traffic.report import (BASE_SUFFIX, BLUE_SCALE, DAY_COLORS, FAMILY_COLORS, INDIGO_SCALE,  # noqa: E402
                                  REFERENCE, SPLIT_COLORS, WEEKDAYS, ablation_summary, best_model_name,
                                  best_ml_name, best_window_start, color_of, comparison_models, density_dots,
                                  export_results, family, final_models, leaderboard, mae_by, plotly_scale,
                                  seed_summary, worst_days, worst_hours)

st.set_page_config(page_title="I-94 Traffic Forecasting (DL)", page_icon="🚦", layout="wide")
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=14, color="#1f2937")


def show(fig, height=None):
    """Render a Plotly figure with one clean, high-contrast style."""
    fig.update_layout(
        template="plotly_white", font=FONT, title_font=dict(size=17, color="#111827"),
        paper_bgcolor="white", plot_bgcolor="white", hoverlabel=dict(font_size=13),
        margin=dict(l=10, r=10, t=60, b=10), legend=dict(bgcolor="rgba(255,255,255,0.9)"))
    fig.update_xaxes(showline=True, linecolor="#9ca3af", gridcolor="#eef0f3", zeroline=False)
    fig.update_yaxes(showline=True, linecolor="#9ca3af", gridcolor="#eef0f3", zeroline=False)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, theme=None)


@st.cache_data(show_spinner="Loading experiment artifacts ...")
def load(out_dir: str, _mtime: float):
    return load_artifacts(out_dir)


def folder_mtime(path):
    try:
        return max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))
    except (FileNotFoundError, ValueError):
        return 0.0


st.sidebar.title("🚦 I-94 Traffic Forecasting")
out_dir = st.sidebar.text_input("Artifacts folder", os.path.join(ROOT, "outputs"))
if not os.path.isdir(out_dir):
    st.error(f"Folder not found: `{out_dir}`. Run `python train.py` or the notebook first, "
             "then copy its `outputs/` folder here.")
    st.stop()
A = load(out_dir, folder_mtime(out_dir))
if A["results"] is None or A["predictions"] is None:
    st.error("`results.csv` / `predictions.npz` are missing. Run the training pipeline first.")
    st.stop()

P = A["predictions"]
PT = P["test"]
BEST = best_model_name(A)
CFG = A["config"] or {}
SEEDS = CFG.get("seeds", [])
LB_TEST, LB_VAL = leaderboard(A, "test"), leaderboard(A, "val")
FINAL = final_models(A)
TIMES = pd.DatetimeIndex(PT["times"])


def line_style(m):
    return dict(color=color_of(m), width=3 if m == BEST else 2,
                dash="solid" if family(m) in ("DL", "ML") else "dash")


PAGES = ["Overview", "Data Explorer", "Model Comparison", "Seeds & Fine-tuning", "Forecast Explorer",
         "Error Analysis", "Ablation", "Explainability", "Training & Tuning"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(f"Selected model (validation): **{BEST}**")
st.sidebar.caption(f"Seeds per model: {', '.join(map(str, SEEDS))}")
st.sidebar.caption(f"Test hours: {len(PT['y']):,} (one-hour-ahead)")
st.sidebar.markdown("---")
st.sidebar.subheader("Save results")
result_dir = st.sidebar.text_input("Result folder", os.path.join(ROOT, "result"))
if st.sidebar.button("Export figures, tables & summary", use_container_width=True):
    with st.spinner("Exporting ..."):
        files = export_results(A, result_dir)
    st.sidebar.success(f"{len(files)} files saved to `{result_dir}`")

# =================================================================== Overview
if page == "Overview":
    st.title("One-hour-ahead traffic forecasting on I-94 — deep learning")
    st.markdown(
        "**Task (shared with the ML part of the group):** forecast the westbound traffic volume of hour *t* "
        "from a 168-hour window of the shared features — traffic history up to *t − 1*, the calendar of *t* and "
        "the weather observed at *t*. Same rows, chronological 70 / 15 / 15 split (from 2016-01-01) and feature "
        "columns as the ML models. Every DL model is trained with several seeds; the reported forecast is the "
        "mean of the seed models after fine-tuning on train + validation.")
    bm = LB_TEST.loc[BEST]
    ref = LB_TEST["MAE"].get(REFERENCE, np.nan)
    sd = seed_summary(A)
    c = st.columns(5)
    c[0].metric("Selected DL model", BEST, "Optuna-tuned" if (A["best"] or {}).get("tuned") else "default settings")
    c[1].metric("Test MAE (veh/h)", f"{bm['MAE']:.1f}",
                f"{100 * (bm['MAE'] / ref - 1):+.1f} % vs {REFERENCE}" if np.isfinite(ref) else None,
                delta_color="inverse")
    c[2].metric("Test RMSE", f"{bm['RMSE']:.1f}")
    c[3].metric("Test sMAPE / MASE", f"{bm['sMAPE']:.2f} % / {bm.get('MASE', np.nan):.3f}")
    if BEST in sd.index:
        c[4].metric("Single-seed test MAE", f"{sd.loc[BEST, 'test_MAE_mean']:.1f} ± {sd.loc[BEST, 'test_MAE_std']:.1f}",
                    f"{len(SEEDS)} seeds")

    st.subheader("Test leaderboard")
    t = LB_TEST.copy()
    t = t.drop(columns=[c_ for c_ in ("family", "variant") if c_ in t])
    t.insert(0, "family", [family(m) for m in t.index])
    st.dataframe(t.style.format(
        {"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}", "sMAPE": "{:.2f}", "MASE": "{:.3f}"}),
        use_container_width=True)
    st.caption("DL rows without a suffix = seed ensemble fine-tuned on train + validation (the final forecast); "
               f"'{BASE_SUFFIX.strip()}' = the same seed models trained on train only. The model was chosen on "
               "validation MAE of the no-fine-tune ensembles; the test set was never used for any choice.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Data splits")
        if A["splits"] is not None:
            st.dataframe(A["splits"], use_container_width=True, hide_index=True)
        d = A["data"] or {}
        st.caption(f"Data source: {d.get('source', '?')}. {len(d.get('numeric_features', []))} numeric features + "
                   f"`weather_main` (embedding). Look-back {d.get('lookback', 168)} h; "
                   f"{d.get('nan_imputed', 0):,} missing cells inside windows filled with the training median.")
    with c2:
        st.subheader("Pipeline (group docs/methodology.md)")
        st.markdown(
            "1. Shared processed data (group contract) → 168-h windows, train-only imputation and scaling\n"
            "2. Baselines: persistence, seasonal naive (t−24), hour-of-week mean\n"
            f"3. Optuna tuning of LSTM and TCN ({CFG.get('n_trials', 10)} trials each, ≤ 25 epochs)\n"
            f"4. Final training of both with seeds {', '.join(map(str, SEEDS))} → ensemble mean\n"
            f"5. Fine-tuning of each seed model on train + val ({CFG.get('finetune_epochs', 8)} epochs, lr/5, cosine)\n"
            "6. Best DL model chosen on validation MAE (before fine-tuning)\n"
            "7. Integrated Gradients + weather occlusion; Diebold–Mariano best ML vs best DL; ablation")
        ml = best_ml_name(A)
        if ml:
            st.caption(f"Best ML model on validation (group predictions): **{ml}**")

# =================================================================== Data Explorer
elif page == "Data Explorer":
    st.title("Data explorer")
    H = A["hourly"]
    if H is None:
        st.info("hourly.csv.gz is missing.")
        st.stop()
    d = H[H["split"] != ""].copy() if (H["split"] != "").any() else H.copy()
    split_sel = st.multiselect("Splits", ["train", "val", "test"], ["train", "val", "test"])
    d = d[d["split"].isin(split_sel)] if "split" in d else d
    daily = d.set_index("date_time").groupby("split")["traffic_volume"].resample("D").mean().reset_index()
    fig = px.line(daily, x="date_time", y="traffic_volume", color="split", color_discrete_map=SPLIT_COLORS,
                  title="Daily mean traffic volume of the modelled hours")
    show(fig, 380)

    c1, c2 = st.columns(2)
    with c1:
        g = d.assign(dow=d["date_time"].dt.dayofweek, hour=d["date_time"].dt.hour)
        hm = g.pivot_table(index="dow", columns="hour", values="traffic_volume", aggfunc="mean")
        hm.index = [WEEKDAYS[i] for i in hm.index]
        fig = px.imshow(hm, color_continuous_scale=plotly_scale(BLUE_SCALE), aspect="auto",
                        title="Mean volume: weekday × hour", labels=dict(color="veh/h"))
        show(fig, 380)
    with c2:
        g["day_type"] = np.select([g["is_holiday"] == 1, g["is_weekend"] == 1], ["Holiday", "Weekend"], "Weekday")
        prof = g.groupby(["day_type", "hour"])["traffic_volume"].mean().reset_index()
        fig = px.line(prof, x="hour", y="traffic_volume", color="day_type", color_discrete_map=DAY_COLORS,
                      markers=True, title="Daily profile by day type")
        show(fig, 380)
    c1, c2 = st.columns(2)
    with c1:
        w = d.groupby("weather_main")["traffic_volume"].agg(["mean", "size"]).sort_values("size", ascending=False)
        fig = px.bar(w.reset_index(), x="weather_main", y="mean", text="size",
                     title="Mean volume by weather_main (label = hours)", color_discrete_sequence=["#1f4e9c"])
        show(fig, 380)
    with c2:
        fig = px.scatter(d.sample(min(len(d), 4000), random_state=0), x="temp_c", y="traffic_volume", opacity=0.35,
                         title="Temperature vs volume (sample)", color_discrete_sequence=["#1f4e9c"])
        show(fig, 380)

# =================================================================== Model Comparison
elif page == "Model Comparison":
    st.title("Model comparison")
    split = st.radio("Split", ["test", "val"], horizontal=True)
    lb = LB_TEST if split == "test" else LB_VAL
    metric = st.selectbox("Metric", ["MAE", "RMSE", "sMAPE", "MASE", "R2"])
    r = lb.sort_values(metric, ascending=metric == "R2").reset_index()
    r["family"] = [family(m) for m in r["model"]]
    fig = px.bar(r, y="model", x=metric, color="family", orientation="h", color_discrete_map=FAMILY_COLORS,
                 text=r[metric].round(3 if metric in ("MASE", "R2") else 1), title=f"{metric} on {split}")
    fig.update_yaxes(categoryorder="array", categoryarray=r["model"][::-1])
    show(fig, 120 + 32 * len(r))
    if split == "val":
        st.caption("On validation only the train-only DL ensembles exist (fine-tuning uses the validation data).")

    st.subheader("Diebold–Mariano tests (test set)")
    if A["dm"] is not None:
        dm = A["dm"].copy()
        st.dataframe(dm.style.format({"mean_d": "{:.2f}", "dm_stat": "{:.2f}", "p_value": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("d_t = |error of A| − |error of B|; mean_d < 0 means A is more accurate. Newey–West variance "
                   "with 24 lags, two-sided p-value. 'better' is only filled when p < 0.05.")
    else:
        st.info("No DM tests saved.")

# =================================================================== Seeds & fine-tuning
elif page == "Seeds & Fine-tuning":
    st.title("Seeds, ensembles and fine-tuning")
    S, sd = A["seeds"], seed_summary(A)
    if S is None:
        st.info("seed_results.csv is missing.")
        st.stop()
    st.markdown(f"Every encoder is trained with seeds **{', '.join(map(str, SEEDS))}**. The single runs show how much "
                "the result depends on the random initialisation; the forecast used is the mean of the seed models.")
    col = st.radio("Metric", ["val_MAE", "test_MAE", "test_MAE_no_finetune"], horizontal=True,
                   format_func=lambda c: {"val_MAE": "Validation MAE", "test_MAE": "Test MAE (fine-tuned)",
                                          "test_MAE_no_finetune": "Test MAE (train only)"}[c])
    ens_col = {"val_MAE": "val_MAE_ensemble", "test_MAE": "test_MAE_ensemble",
               "test_MAE_no_finetune": "test_MAE_ensemble_no_finetune"}[col]
    fig = go.Figure()
    for m in sd.index:
        v = S[S["model"] == m]
        fig.add_trace(go.Scatter(x=v[col], y=[m] * len(v), mode="markers", name=m, marker=dict(size=11, color=color_of(m)),
                                 text=[f"seed {s}" for s in v["seed"]], hovertemplate="%{text}: %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sd[ens_col], y=sd.index, mode="markers", name="seed ensemble",
                             marker=dict(symbol="diamond", size=15, color="black")))
    fig.update_layout(title="Single seeds (dots) vs ensemble (diamond)", xaxis_title="MAE (vehicles/hour)")
    show(fig, 120 + 60 * len(sd))
    st.dataframe(sd.style.format({c: "{:.1f}" for c in sd.columns if "MAE" in c or c == "epochs_mean"}
                                 | {"seconds_total": "{:.0f}", "n_params": "{:,}"}), use_container_width=True)

    st.subheader("Effect of fine-tuning on train + validation")
    f = sd[["test_MAE_ensemble_no_finetune", "test_MAE_ensemble"]].rename(
        columns={"test_MAE_ensemble_no_finetune": "train only", "test_MAE_ensemble": "fine-tuned"}).reset_index()
    f = f.melt(id_vars="model", var_name="variant", value_name="test MAE")
    fig = px.bar(f, x="model", y="test MAE", color="variant", barmode="group", text=f["test MAE"].round(1),
                 color_discrete_map={"train only": FAMILY_COLORS["DL (no fine-tune)"], "fine-tuned": FAMILY_COLORS["DL"]})
    lo, hi = f["test MAE"].min(), f["test MAE"].max()
    fig.update_yaxes(range=[max(0, lo - 0.3 * (hi - lo + 10)), hi + 0.3 * (hi - lo + 10)])
    show(fig, 420)

    if "seeds" in P:
        st.subheader(f"{BEST}: single seeds vs ensemble on test")
        rows = [dict(run=k, MAE=float(np.mean(np.abs(v - PT["y"])))) for k, v in P["seeds"].items()]
        rows.append(dict(run="ensemble", MAE=float(np.mean(np.abs(PT["preds"][BEST] - PT["y"])))))
        st.dataframe(pd.DataFrame(rows).style.format({"MAE": "{:.2f}"}), hide_index=True)

# =================================================================== Forecast Explorer
elif page == "Forecast Explorer":
    st.title("Forecast explorer (test period)")
    models = st.multiselect("Models", list(PT["preds"]), comparison_models(A))
    start_default = best_window_start(TIMES, 7).date()
    start = st.date_input("Start date", start_default, min_value=TIMES.min().date(), max_value=TIMES.max().date())
    days = st.slider("Days", 1, 28, 7)
    s = pd.Timestamp(start)
    full = pd.date_range(s, s + pd.Timedelta(days=days), freq="h", inclusive="left")
    pos = pd.Series(np.arange(len(TIMES)), index=TIMES).reindex(full)
    ok = pos.notna().to_numpy()
    ii = pos.fillna(0).astype(int).to_numpy()

    def seg(v):
        return np.where(ok, np.asarray(v, float)[ii], np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full, y=seg(PT["y"]), name="Actual", line=dict(color="black", width=3)))
    for m in models:
        fig.add_trace(go.Scatter(x=full, y=seg(PT["preds"][m]), name=m, line=line_style(m)))
    fig.update_layout(title="One-hour-ahead forecasts (gaps = hours without a full 168-h history)",
                      yaxis_title="vehicles/hour", legend=dict(orientation="h", y=-0.15))
    show(fig, 480)
    if ok.any() and models:
        sel = ii[ok]
        st.dataframe(pd.DataFrame({m: [float(np.mean(np.abs(PT["preds"][m][sel] - PT["y"][sel])))] for m in models},
                                  index=["MAE in this window"]).style.format("{:.1f}"), use_container_width=True)
    else:
        st.info("No test hours in the selected window.")

# =================================================================== Error Analysis
elif page == "Error Analysis":
    st.title("Error analysis (test)")
    models = st.multiselect("Models", FINAL, comparison_models(A))
    if not models:
        st.stop()
    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "hour", "MAE by hour of day"), (c2, "dow", "MAE by day of week"),
                            (c1, "day_type", "MAE by day type"), (c2, "month", "MAE by month"),
                            (c1, "weather", "MAE by weather_main at the target hour")]:
        d = mae_by(A, key, models).reset_index().rename(columns={"index": key})
        long = d.melt(id_vars=[key, "n"], value_vars=models, var_name="model", value_name="MAE")
        with col:
            if key == "hour":
                fig = px.line(long, x=key, y="MAE", color="model", markers=True, title=title,
                              color_discrete_map={m: color_of(m) for m in models})
            else:
                fig = px.bar(long, x=key, y="MAE", color="model", barmode="group", title=title,
                             hover_data=["n"], color_discrete_map={m: color_of(m) for m in models})
            show(fig, 380)
    with c2:
        m = st.selectbox("Model for the scatter", models, index=0)
        y, p = PT["y"], PT["preds"][m]
        lim = float(max(y.max(), p.max()) * 1.03)
        cx, cy, cnt = density_dots(y, p, lim)
        fig = go.Figure(go.Scatter(x=cx, y=cy, mode="markers",
                                   marker=dict(color=np.log10(cnt), colorscale=plotly_scale(BLUE_SCALE[1:]), size=6,
                                               colorbar=dict(title="log10 hours"))))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color="black", dash="dash"),
                                 showlegend=False))
        fig.update_layout(title=f"{m}: forecast vs actual", xaxis_title="actual", yaxis_title="forecast",
                          showlegend=False)
        show(fig, 420)
    st.subheader(f"Largest errors of {BEST}")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(worst_hours(A, BEST).style.format({"actual": "{:.0f}", "forecast": "{:.0f}", "error": "{:+.0f}"}),
                     hide_index=True, use_container_width=True)
    with c2:
        st.dataframe(worst_days(A, BEST).style.format({"MAE": "{:.1f}"}), use_container_width=True)

# =================================================================== Ablation
elif page == "Ablation":
    st.title("Feature-group ablation")
    ab = ablation_summary(A)
    if not len(ab):
        st.info("No ablation results (run without --no-ablation).")
        st.stop()
    st.markdown(f"Cumulative groups for **{BEST}** with the selected hyper-parameters, trained on train only. "
                "Validation is the primary metric, test is reported for reference. Without the weather group the "
                "`weather_main` embedding is removed as well.")
    fig = go.Figure()
    for col, std, name, color in [("val_MAE", "val_std", "validation", "#1f4e9c"), ("test_MAE", "test_std", "test", "#e07b39")]:
        fig.add_trace(go.Scatter(x=ab["features"], y=ab[col], error_y=dict(type="data", array=ab[std]),
                                 mode="lines+markers", name=name, line=dict(color=color, width=3)))
    raw = A["ablation"]
    lbl = dict(zip(ab["stage"], ab["features"]))
    fig.add_trace(go.Scatter(x=raw["stage"].map(lbl), y=raw["val_MAE"], mode="markers", name="single seeds (val)",
                             marker=dict(color="#1f4e9c", size=7, opacity=0.4)))
    fig.update_layout(title="MAE (mean ± std over seeds)", yaxis_title="MAE (vehicles/hour)")
    show(fig, 460)
    st.dataframe(ab.style.format({c: "{:.1f}" for c in ab.columns if "MAE" in c or "std" in c or "gain" in c}),
                 hide_index=True, use_container_width=True)

# =================================================================== Explainability
elif page == "Explainability":
    st.title(f"Explaining {BEST} (fine-tuned seed ensemble, test windows)")
    igf, igl, igm = A["ig_features"], A["ig_lag"], A["ig_matrix"]
    if igf is not None:
        c1, c2 = st.columns(2)
        with c1:
            n = st.slider("Features shown", 5, len(igf), min(20, len(igf)))
            top = igf.head(n).iloc[::-1]
            fig = px.bar(top, x="mean_abs_attr", y="feature", orientation="h", color_discrete_sequence=["#1f4e9c"],
                         title="Integrated Gradients by feature (mean |IG|, standardised units)")
            show(fig, 120 + 24 * n)
        with c2:
            fig = px.line(igl, x="lag", y="mean_abs_ig",
                          title="Attribution over the 168-hour window (lag 0 = row of hour t)")
            show(fig, 420)
        if igm is not None:
            feats = st.multiselect("Features in the heat map", list(igm.columns), igf["feature"].head(10).tolist())
            if feats:
                fig = px.imshow(igm[feats].T, aspect="auto", color_continuous_scale=plotly_scale(INDIGO_SCALE),
                                labels=dict(x="lag", y="feature", color="mean |IG|"),
                                title="|IG| by feature and position")
                show(fig, 140 + 28 * len(feats))
        st.caption("256 random test windows, 24 steps, zero baseline in standardised space (= training mean). "
                   "IG is linear in the output, so the ensemble attribution is the mean of the per-seed attributions.")
    if A["occlusion"] is not None and len(A["occlusion"]):
        o = A["occlusion"]
        fig = px.bar(o, x="weather", y="mean_abs_delta", hover_data=["windows_with_category", "mean_abs_delta_scaled"],
                     color_discrete_sequence=["#2ca02c"],
                     title="Weather occlusion: mean |change of the prediction| when a category is hidden (veh/h)")
        show(fig, 380)

# =================================================================== Training & Tuning
elif page == "Training & Tuning":
    st.title("Training and tuning")
    b = A["best"] or {}
    st.subheader("Selected configuration")
    st.json({k: b.get(k) for k in ("model", "tuned", "hp", "seeds", "finetune_epochs", "n_params", "selection")})
    if A["tuning"] is not None:
        st.subheader("Tuning trials")
        tl = A["tuning"]
        fig = px.scatter(tl, x="trial", y="val_MAE", color="model", symbol="model", title="Optuna: validation MAE per trial (seed 42, ≤ 25 epochs)",
                         color_discrete_map={m: color_of(m) for m in tl["model"].unique()})
        show(fig, 380)
        st.dataframe(tl, hide_index=True, use_container_width=True)
    H = A["histories"] or {}
    runs = sorted(k for k in H if "fine-tune" not in k)
    if runs:
        st.subheader("Learning curves")
        kinds = sorted({k.split(" | ")[0] for k in runs})
        kind = st.selectbox("Model", kinds, index=kinds.index(BEST) if BEST in kinds else 0)
        c1, c2 = st.columns(2)
        f1, f2 = go.Figure(), go.Figure()
        for k in runs:
            if k.startswith(kind + " | "):
                h = H[k]
                f1.add_trace(go.Scatter(x=h["epoch"], y=h["val_mae"], name=k.split(" | ")[1], mode="lines"))
                f2.add_trace(go.Scatter(x=h["epoch"], y=h["lr"], name=k.split(" | ")[1], mode="lines"))
        f1.update_layout(title="Validation MAE per epoch", xaxis_title="epoch")
        f2.update_layout(title="Learning rate (ReduceLROnPlateau)", xaxis_title="epoch", yaxis_type="log")
        with c1:
            show(f1, 380)
        with c2:
            show(f2, 380)
        ft = {k: v for k, v in H.items() if k.startswith(kind + " | ") and "fine-tune" in k}
        if ft:
            f3 = go.Figure()
            for k, h in ft.items():
                f3.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name=k.split(" | ")[1], mode="lines+markers"))
            f3.update_layout(title="Fine-tuning loss on train + val (SmoothL1, scaled target)", xaxis_title="epoch")
            show(f3, 340)
    d = (A["data"] or {}).get("timings")
    if d:
        st.caption("Run time (s): " + ", ".join(f"{k} {v:.0f}" for k, v in d.items()))
