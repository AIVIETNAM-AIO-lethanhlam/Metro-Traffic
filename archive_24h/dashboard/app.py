"""Streamlit dashboard for the Metro Interstate Traffic forecasting experiment.

Run from the project root (so that .streamlit/config.toml is picked up):
    streamlit run dashboard/app.py

It only reads the files written to `outputs/` by `train.py` or the notebook,
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
from metro_traffic.report import (BASELINES, BLUE_SCALE, DAY_COLORS, INDIGO_SCALE, RIDGE, SNAIVE,  # noqa: E402
                                  WEEKDAYS, best_model_name, density_dots, plotly_scale, binned_heatmap, color_map, export_results,
                                  mae_by, mae_by_horizon, model_kind, stitched_days, target_index, worst_days)

st.set_page_config(page_title="I-94 Traffic Forecasting", page_icon="🚦", layout="wide")

KIND_COLORS = {"Baseline": "#e07b39", "Deep learning": "#1f4e9c", "Ablation": "#9e9e9e"}
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=14, color="#1f2937")


def show(fig, height=None):
    """Render a Plotly figure with one clean, high-contrast style (white background, solid colours)."""
    fig.update_layout(
        template="plotly_white", font=FONT, title_font=dict(size=17, color="#111827"),
        paper_bgcolor="white", plot_bgcolor="white", hoverlabel=dict(font_size=13),
        margin=dict(l=10, r=10, t=60, b=10), legend=dict(bgcolor="rgba(255,255,255,0.9)"))
    fig.update_xaxes(showline=True, linecolor="#9ca3af", gridcolor="#eef0f3", zeroline=False)
    fig.update_yaxes(showline=True, linecolor="#9ca3af", gridcolor="#eef0f3", zeroline=False)
    if height:
        fig.update_layout(height=height)
    # theme=None: use our template instead of Streamlit's own (which looked washed out)
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------------------------------------------------------------- loading
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
    st.error("`results_test.csv` / `predictions_test.npz` are missing. Run the training pipeline first.")
    st.stop()

RES = A["results"]
PRED = A["predictions"]
MODEL_NAMES = list(PRED["preds"])
BEST = best_model_name(A)
COLOR = color_map(A)
Y = PRED["y_true"]
TT = PRED["target_times"]
ORIGIN_TIMES = pd.DatetimeIndex(TT[:, 0]) - pd.Timedelta(hours=1)
ORIGIN_POS = {t: i for i, t in enumerate(ORIGIN_TIMES)}
H = Y.shape[1]


def mae(a, b, axis=None):
    return np.mean(np.abs(np.asarray(a) - np.asarray(b)), axis=axis)


def line_style(m):
    return dict(color=COLOR.get(m, "#444"), width=3 if m == BEST else 2, dash="dash" if m in BASELINES else "solid")


page = st.sidebar.radio("Page", ["Overview", "Data Explorer", "Model Comparison", "Forecast Explorer",
                                 "Error Analysis", "Explainability", "Training & Tuning"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Best model (selected on validation): **{BEST}**")
st.sidebar.caption(f"Test windows: {len(Y):,} × {H} hours")

st.sidebar.markdown("---")
st.sidebar.subheader("Save results")
result_dir = st.sidebar.text_input("Result folder", os.path.join(ROOT, "result"))
if st.sidebar.button("Export figures, tables & summary", use_container_width=True):
    with st.spinner("Exporting ..."):
        files = export_results(A, result_dir)
    st.sidebar.success(f"{len(files)} files saved to `{result_dir}`")

# =================================================================== Overview
if page == "Overview":
    st.title("Hourly Traffic Volume Forecasting on I-94 with Deep Learning")
    st.markdown(
        "**Task.** At each forecast origin *t*, use the previous **168 hours** of traffic, weather and calendar "
        "data, plus the **known calendar of the next 24 hours**, to forecast westbound traffic volume on "
        "Interstate 94 (Minneapolis–St Paul) for hours *t+1 … t+24*.  \n"
        "**Data.** UCI *Metro Interstate Traffic Volume*, Oct 2012 – Sep 2018, hourly.  \n"
        "**Split (chronological).** Train ≤ 2016 · Validation = 2017 · Test = Jan–Sep 2018.")

    best = RES.loc[BEST]
    c = st.columns(5)
    c[0].metric("Best model", BEST.replace("DL-", ""))
    c[1].metric("Test MAE (veh/h)", f"{best.MAE:,.1f}")
    c[2].metric("Test R²", f"{best.R2:.3f}")
    if SNAIVE in RES.index:
        c[3].metric("vs Seasonal-naive", f"{100 * (1 - best.MAE / RES.loc[SNAIVE, 'MAE']):.1f}% lower MAE")
    if RIDGE in RES.index:
        c[4].metric("vs Ridge", f"{100 * (1 - best.MAE / RES.loc[RIDGE, 'MAE']):.1f}% lower MAE")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Test leaderboard")
        tbl = RES[[c for c in ["MAE", "RMSE", "sMAPE", "R2", "rMAE"] if c in RES.columns]].copy()
        tbl.insert(0, "type", [model_kind(m) for m in tbl.index])
        st.dataframe(tbl.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "sMAPE": "{:.2f}", "R2": "{:.4f}",
                                       "rMAE": "{:.3f}"}).background_gradient(subset=["MAE"], cmap="RdYlGn_r"),
                     use_container_width=True)
        st.caption("rMAE = model MAE / seasonal-naive MAE (below 1 means better than 'same hour last week').")
    with right:
        st.subheader("Data splits")
        if A["splits"] is not None:
            st.dataframe(A["splits"], hide_index=True, use_container_width=True)
        st.subheader("Data quality (raw file)")
        if A["quality"]:
            q = A["quality"]
            st.markdown(
                f"- {q['rows']:,} rows, {q['unique_timestamps']:,} unique hours\n"
                f"- {q['duplicate_timestamps']:,} duplicated timestamps (several weather labels per hour → multi-hot)\n"
                f"- `holiday` {q['holiday_missing_pct']}% empty; labels only on hour(s) {q['holiday_label_hours']} → spread to whole day\n"
                f"- {q['temp_zero_kelvin']} rows with temp = 0 K, {q['rain_over_100mm']} with rain > 100 mm → treated as sensor errors")

    st.subheader("Pipeline")
    st.markdown(
        "1. **Cleaning** — deduplicate timestamps, fix sensor errors, reindex to an hourly grid, interpolate gaps "
        "≤ 3 h and exclude long gaps (e.g. the 307-day outage).\n"
        "2. **Features** — scaled traffic, weather (log-rain/snow, temperature, clouds, weather multi-hot), "
        "cyclical hour/day/season, weekend/holiday flags; scalers fitted on train only.\n"
        "3. **Baselines** — daily-naive, seasonal-naive (weekly), Ridge on 168 lags.\n"
        "4. **Deep learning** — MLP, LSTM, GRU, TCN and PatchTransformer encoders sharing a head that sees the "
        "future calendar and a learned weekly-seasonal skip; Huber loss, AdamW, LR scheduling, early stopping.\n"
        "5. **Tuning & ablation** — random search on validation; retrain without weather features.\n"
        "6. **Explanation** — permutation importance and Integrated Gradients.")

# ============================================================== Data Explorer
elif page == "Data Explorer":
    st.title("Data Explorer")
    hd = A["hourly"]
    if hd is None:
        st.warning("`hourly_clean.csv.gz` not found in the artifacts folder.")
        st.stop()
    hd = hd[hd.valid == 1].copy()
    hd["hour"] = hd.date_time.dt.hour
    hd["weekday"] = hd.date_time.dt.dayofweek
    hd["day_type"] = np.select([hd.is_holiday == 1, hd.is_weekend == 1], ["Holiday", "Weekend"], "Weekday")

    dmin, dmax = hd.date_time.min().date(), hd.date_time.max().date()
    c1, c2 = st.columns([3, 1])
    rng = c1.slider("Date range", min_value=dmin, max_value=dmax,
                    value=(pd.Timestamp("2018-06-01").date(), pd.Timestamp("2018-06-14").date()))
    res = c2.selectbox("Resolution", ["Hourly", "Daily mean", "Weekly mean"], index=0)
    sub = hd[(hd.date_time.dt.date >= rng[0]) & (hd.date_time.dt.date <= rng[1])].set_index("date_time")
    if res != "Hourly":
        sub = sub[["traffic_volume", "temp_c"]].resample("D" if res == "Daily mean" else "W").mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub.index, y=sub.traffic_volume, name="Traffic volume",
                             line=dict(color="#1f4e9c", width=2)))
    fig.add_trace(go.Scatter(x=sub.index, y=sub.temp_c, name="Temperature (°C)", yaxis="y2",
                             line=dict(width=1.6, color="#e07b39", dash="dot")))
    fig.update_layout(title="Traffic volume and temperature", yaxis=dict(title="vehicles / hour"),
                      yaxis2=dict(title="°C", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12))
    show(fig, 400)

    c1, c2 = st.columns(2)
    with c1:
        piv = hd.pivot_table(index="weekday", columns="hour", values="traffic_volume", aggfunc="mean") / 1000
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=list(range(24)), y=WEEKDAYS, colorscale=plotly_scale(BLUE_SCALE), xgap=2, ygap=2,
            text=piv.values, texttemplate="%{text:.1f}", textfont=dict(size=11),
            colorbar=dict(title="k veh/h"),
            hovertemplate="%{y} %{x}:00<br>%{z:.2f}k vehicles/hour<extra></extra>"))
        fig.update_layout(title="Average traffic (thousand vehicles/hour): weekday × hour",
                          xaxis=dict(title="Hour of day", dtick=1, showgrid=False),
                          yaxis=dict(autorange="reversed", showgrid=False))
        show(fig, 420)
    with c2:
        prof = hd.groupby(["day_type", "hour"]).traffic_volume.mean().reset_index()
        fig = px.line(prof, x="hour", y="traffic_volume", color="day_type", markers=True,
                      color_discrete_map=DAY_COLORS, category_orders={"day_type": list(DAY_COLORS)},
                      title="Hourly profile by day type (holidays behave like weekends)",
                      labels=dict(traffic_volume="vehicles / hour", hour="Hour of day", day_type=""))
        fig.update_traces(line=dict(width=3))
        show(fig, 420)

    c1, c2 = st.columns(2)
    with c1:
        day = hd[(hd.hour >= 6) & (hd.hour <= 19) & (hd.day_type == "Weekday")]
        order = day.groupby("weather_main").traffic_volume.median().sort_values().index.tolist()
        fig = px.box(day, x="weather_main", y="traffic_volume", category_orders={"weather_main": order},
                     points=False, title="Weekday daytime (6–19h) traffic by weather",
                     labels=dict(weather_main="", traffic_volume="vehicles / hour"))
        fig.update_traces(marker_color="#1f4e9c", line_color="#1f4e9c", fillcolor="rgba(31,78,156,0.25)")
        show(fig, 420)
    with c2:
        fig = px.histogram(hd, x="traffic_volume", nbins=70, title="Distribution of hourly traffic (bimodal: night vs day)",
                           labels=dict(traffic_volume="vehicles / hour"))
        fig.update_traces(marker_color="#1f4e9c", marker_line_color="white", marker_line_width=0.5)
        show(fig, 420)

    with st.expander("Holidays found in the data"):
        hol = hd.dropna(subset=["holiday_name"]).groupby(hd.date_time.dt.date).holiday_name.first()
        st.dataframe(hol.rename("holiday").to_frame(), use_container_width=True)

# =========================================================== Model Comparison
elif page == "Model Comparison":
    st.title("Model Comparison (test set, Jan–Sep 2018)")
    metric = st.radio("Metric", ["MAE", "RMSE", "sMAPE", "R2", "rMAE"], horizontal=True)
    df = RES.reset_index().rename(columns={"index": "model"})
    df["type"] = [model_kind(m) for m in df.model]
    lower_better = metric != "R2"
    df = df.sort_values(metric, ascending=not lower_better)
    fig = px.bar(df, x=metric, y="model", color="type", orientation="h", color_discrete_map=KIND_COLORS,
                 text=df[metric].map(lambda v: f"{v:.3f}" if metric in ("R2", "rMAE") else f"{v:.1f}"),
                 title=f"{metric} by model" + (" (lower is better)" if lower_better else " (higher is better)"))
    fig.update_traces(textposition="outside", cliponaxis=False)
    if metric == "rMAE":
        fig.add_vline(x=1, line_dash="dash", line_color="#111", annotation_text="seasonal-naive")
    fig.update_layout(yaxis_title="", legend_title="")
    show(fig, 90 + 42 * len(df))

    st.subheader("Error by forecast horizon")
    default = [m for m in [SNAIVE, RIDGE, BEST] if m in MODEL_NAMES]
    sel = st.multiselect("Models", MODEL_NAMES, default=default)
    if sel:
        hz = mae_by_horizon(A, sel)
        fig = go.Figure()
        for m in sel:
            fig.add_trace(go.Scatter(x=hz.index, y=hz[m], name=m, mode="lines+markers", line=line_style(m),
                                     marker=dict(size=6)))
        fig.update_layout(title="MAE by forecast step", xaxis=dict(title="Hours ahead", dtick=2),
                          yaxis_title="MAE (vehicles / hour)")
        show(fig, 430)
        st.caption("Errors usually grow with the horizon: the next hour is easier to forecast than 24 hours ahead.")

    if {"params", "train_sec"} <= set(RES.columns):
        d = RES.dropna(subset=["params", "train_sec"]).reset_index().rename(columns={"index": "model"})
        if len(d):
            st.subheader("Accuracy vs. model size and training time")
            fig = px.scatter(d, x="train_sec", y="MAE", size=np.sqrt(d.params.astype(float)), color="model",
                             hover_data=["params"], color_discrete_map=COLOR, text="model",
                             labels=dict(train_sec="Training time (s)", MAE="Test MAE"))
            fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white"), opacity=0.9))
            fig.update_layout(showlegend=False, title="Bubble size = number of parameters")
            show(fig, 450)

    st.subheader("Full results table")
    st.dataframe(RES.style.format(precision=3, na_rep="—"), use_container_width=True)
    st.download_button("Download results (CSV)", RES.to_csv().encode(), "results_test.csv", "text/csv")

# ========================================================== Forecast Explorer
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    default = [m for m in [BEST, SNAIVE] if m in MODEL_NAMES]
    sel = st.multiselect("Models to show", MODEL_NAMES, default=default)
    mode = st.radio("View", ["Single 24-hour forecast", "Day-ahead forecasts stitched over several days"],
                    horizontal=True)
    tmin, tmax = ORIGIN_TIMES.min(), ORIGIN_TIMES.max()

    if mode == "Single 24-hour forecast":
        c1, c2 = st.columns(2)
        d = c1.date_input("Origin date", value=pd.Timestamp("2018-05-27").date(),
                          min_value=tmin.date(), max_value=tmax.date())
        hr = c2.slider("Origin hour (last observed hour)", 0, 23, 23)
        t = pd.Timestamp(d) + pd.Timedelta(hours=hr)
        if t not in ORIGIN_POS:
            st.warning("No complete window for this origin (data gap). Pick another time.")
            st.stop()
        i = ORIGIN_POS[t]
        x = pd.DatetimeIndex(TT[i])
        fig = go.Figure(go.Scatter(x=x, y=Y[i], name="Actual", mode="lines+markers",
                                   line=dict(color="#111", width=3.5), marker=dict(size=6)))
        rows = []
        for m in sel:
            fig.add_trace(go.Scatter(x=x, y=PRED["preds"][m][i], name=m, mode="lines", line=line_style(m)))
            rows.append(dict(model=m, MAE=mae(PRED["preds"][m][i], Y[i])))
        fig.update_layout(title=f"Forecast made at {t:%Y-%m-%d %H:%M} (next day: {PRED['day_type'][i][-1]})",
                          yaxis_title="vehicles / hour", legend=dict(orientation="h", y=-0.15))
        show(fig, 460)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.1f}"), use_container_width=True)
    else:
        c1, c2 = st.columns(2)
        start = c1.date_input("First day", value=pd.Timestamp("2018-05-26").date(),
                              min_value=(tmin + pd.Timedelta(days=1)).date(), max_value=tmax.date())
        days = c2.slider("Number of days", 1, 21, 7)
        x, idx = stitched_days(A, start, days)
        if not idx:
            st.warning("No complete windows in this period.")
            st.stop()
        fig = go.Figure(go.Scatter(x=x, y=Y[idx].ravel(), name="Actual", line=dict(color="#111", width=3)))
        for m in sel:
            fig.add_trace(go.Scatter(x=x, y=PRED["preds"][m][idx].ravel(), name=m, line=line_style(m)))
        for k in range(days):
            day0 = pd.Timestamp(start) + pd.Timedelta(days=k)
            if day0.dayofweek >= 5:
                fig.add_vrect(x0=day0, x1=day0 + pd.Timedelta(days=1), fillcolor="#9ca3af", opacity=0.15, line_width=0)
        fig.update_layout(title="Each day is forecast at 23:00 of the previous day (shaded = weekend)",
                          yaxis_title="vehicles / hour", legend=dict(orientation="h", y=-0.15))
        show(fig, 480)
        st.dataframe(pd.DataFrame({m: [mae(PRED["preds"][m][idx], Y[idx])] for m in sel}, index=["MAE"])
                     .style.format("{:.1f}"), use_container_width=True)

# ============================================================== Error Analysis
elif page == "Error Analysis":
    st.title("Error Analysis")
    c1, c2 = st.columns(2)
    m = c1.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(BEST))
    ref = c2.multiselect("Compare with", [o for o in MODEL_NAMES if o != m],
                         default=[o for o in [SNAIVE, RIDGE] if o in MODEL_NAMES and o != m])
    models = [m] + ref

    c1, c2 = st.columns(2)
    with c1:
        g = mae_by(A, "hour", models)
        fig = go.Figure([go.Scatter(x=g.index, y=g[k], name=k, mode="lines+markers", line=line_style(k)) for k in models])
        fig.update_layout(title="MAE by hour of day", xaxis=dict(title="Hour of day", dtick=2), yaxis_title="MAE",
                          legend=dict(orientation="h", y=-0.2))
        show(fig, 420)
    with c2:
        g = mae_by(A, "day_type", models)
        fig = go.Figure([go.Bar(x=g.index, y=g[k], name=k, marker_color=COLOR.get(k), text=g[k].round(0),
                                textposition="outside") for k in models])
        fig.update_layout(title="MAE by day type", barmode="group", yaxis_title="MAE",
                          legend=dict(orientation="h", y=-0.2))
        show(fig, 420)

    # ---- predicted vs actual: own full-width row so the dots have room
    st.subheader(f"Predicted vs actual — {m}")
    y_all, p_all = Y.ravel(), PRED["preds"][m].ravel()
    lim = float(max(y_all.max(), p_all.max()))
    c1, c2, c3 = st.columns([2, 1, 1])
    view = c1.radio("View", ["Density (all hours)", "Scatter by day type (sample)"], horizontal=True)
    if view.startswith("Density"):
        bins = c2.select_slider("Grid resolution", options=[60, 90, 120, 160, 200], value=160)
        dot = c3.select_slider("Dot size", options=[3, 4, 5, 6, 8], value=4)
        dx, dy, dc = density_dots(y_all, p_all, lim, bins=bins)
        ticks = [t for t in [1, 10, 100, 1000, 10000] if t <= dc.max()]
        fig = go.Figure(go.Scattergl(
            x=dx, y=dy, mode="markers", name="hours", showlegend=False, customdata=dc,
            marker=dict(size=dot, symbol="circle", color=np.log10(dc), colorscale=plotly_scale(BLUE_SCALE[1:]),
                        cmin=0, line=dict(width=0),
                        colorbar=dict(title="hours", tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks])),
            hovertemplate="actual ≈ %{x:.0f}<br>predicted ≈ %{y:.0f}<br>%{customdata:.0f} hours<extra></extra>"))
    else:
        n = c2.select_slider("Sample size", options=[1000, 2000, 3000, 5000, 8000], value=3000)
        dot = c3.select_slider("Dot size", options=[3, 4, 5, 6, 8], value=5)
        base = target_index(A).assign(pred=p_all).sample(n, random_state=0)
        fig = go.Figure([go.Scattergl(x=g.y, y=g.pred, mode="markers", name=d,
                                      marker=dict(size=dot, color=DAY_COLORS[d], opacity=0.65, line=dict(width=0)))
                         for d, g in base.groupby("day_type")])
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect forecast",
                             line=dict(color="#d62728", dash="dash", width=2)))
    # no fixed 1:1 aspect ratio -> the plot stretches over the full page width
    fig.update_layout(xaxis=dict(title="Actual (veh/h)", range=[0, lim * 1.02], dtick=500),
                      yaxis=dict(title="Predicted (veh/h)", range=[0, lim * 1.02], dtick=1000),
                      legend=dict(orientation="h", y=-0.12))
    show(fig, 620)

    # ---- residuals: separate row
    resid = p_all - y_all
    fig = go.Figure(go.Histogram(x=resid, xbins=dict(start=-2500, end=2500, size=50),
                                 marker=dict(color="#1f4e9c", line=dict(width=0.5, color="white"))))
    fig.add_vline(x=0, line_color="#d62728", line_width=2)
    fig.update_layout(title=f"Residuals of {m} (mean {resid.mean():.1f}, std {resid.std():.0f})",
                      xaxis=dict(title="prediction − actual (veh/h)", range=[-2500, 2500]), yaxis_title="count")
    show(fig, 420)

    g = mae_by(A, "month", models)
    fig = go.Figure([go.Scatter(x=g.index, y=g[k], name=k, mode="lines+markers", line=line_style(k)) for k in models])
    fig.update_layout(title="MAE by month (2018)", xaxis=dict(title="Month", dtick=1), yaxis_title="MAE")
    show(fig, 380)

    st.subheader("Days with the largest errors")
    st.dataframe(worst_days(A, m).style.format({"MAE": "{:.1f}", "mean_actual": "{:.0f}", "mean_pred": "{:.0f}"}),
                 use_container_width=True)
    st.caption("Large errors usually come from events the model cannot see: storms, accidents, road works, "
               "special events, or holidays such as July 4th that fall mid-week.")

# =============================================================== Explainability
elif page == "Explainability":
    st.title(f"Model Explanation — {BEST}")
    if A["perm"] is None and A["ig_lag"] is None:
        st.info("No explanation artifacts found (they are produced by the deep learning run).")
        st.stop()
    if A["perm"] is not None:
        p = A["perm"].sort_values("delta_MAE")
        fig = go.Figure(go.Bar(x=p.delta_MAE, y=p.group, orientation="h", marker_color="#1f4e9c",
                               error_x=dict(type="data", array=p["std"], color="#111"),
                               text=p.delta_MAE.round(0), textposition="outside", cliponaxis=False))
        fig.update_layout(title="Permutation feature importance",
                          xaxis_title="Increase in test MAE when the group is shuffled (veh/h)")
        show(fig, 420)
        st.caption(f"Base MAE = {p.base_MAE.iloc[0]:.1f}. A larger increase means the model relies more on that group.")

    c1, c2 = st.columns([3, 2])
    if A["ig_lag"] is not None:
        with c1:
            lag = A["ig_lag"].sort_values("hours_before_origin")
            win = st.select_slider("Smoothing window (hours)", options=[1, 3, 6], value=3)
            smooth = lag.importance.rolling(win, center=True, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lag.hours_before_origin, y=lag.importance, name="raw",
                                     line=dict(color="#c6dbef", width=1)))
            fig.add_trace(go.Scatter(x=lag.hours_before_origin, y=smooth, name=f"{win}-hour average",
                                     line=dict(color="#1f4e9c", width=3)))
            for d in range(24, 169, 24):
                fig.add_vline(x=d, line_dash="dot", line_color="#9ca3af",
                              annotation_text=f"{d // 24}d", annotation_position="top")
            fig.update_layout(title="Integrated Gradients by time lag",
                              xaxis=dict(title="Hours before the forecast origin", autorange="reversed", dtick=24),
                              yaxis=dict(title="mean |IG|", rangemode="tozero"),
                              legend=dict(orientation="h", y=-0.2))
            show(fig, 470)
            st.caption("Peaks near 0 h (recent momentum) and around 1 week ago (weekly seasonality) show which "
                       "past hours the network uses.")
    if A["ig_features"] is not None:
        with c2:
            f = A["ig_features"].sort_values("importance")
            fig = go.Figure(go.Bar(x=f.importance, y=f.feature, orientation="h", marker_color="#1f4e9c",
                                   text=f.importance.round(2), textposition="outside", cliponaxis=False))
            fig.update_layout(title="Integrated Gradients by feature", xaxis_title="sum of mean |IG|")
            show(fig, max(470, 24 * len(f) + 120))
    if A["ig_heatmap"] is not None:
        st.subheader("Traffic-history attribution for each forecast step")
        c1, c2 = st.columns(2)
        bin_h = c1.select_slider("Lag bin size (hours)", options=[1, 2, 3, 6, 12], value=3)
        norm = c2.toggle("Normalise each row", value=True)
        hm = binned_heatmap(A["ig_heatmap"], bin_h, norm)
        fig = go.Figure(go.Heatmap(z=hm.values, x=[str(c) for c in hm.columns], y=list(hm.index),
                                   colorscale=plotly_scale(INDIGO_SCALE), xgap=1, ygap=2,
                                   colorbar=dict(title="relative |IG|" if norm else "|IG|"),
                                   hovertemplate="%{x} h before origin<br>%{y}<br>%{z:.3f}<extra></extra>"))
        fig.update_layout(xaxis=dict(title=f"Hours before origin ({bin_h}-hour bins)", type="category",
                                     nticks=15, showgrid=False),
                          yaxis=dict(title="Forecast step", autorange="reversed", showgrid=False))
        show(fig, 430)

# ============================================================ Training & Tuning
elif page == "Training & Tuning":
    st.title("Training & Hyperparameter Tuning")
    hist = A["histories"] or {}
    if hist:
        keys = st.multiselect("Runs", list(hist), default=[k for k in hist if not k.endswith("-tuned")
                                                            and not k.startswith("ablation")])
        rows = [h.assign(run=k) for k, h in hist.items() if k in keys]
        if rows:
            d = pd.concat(rows)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(d, x="epoch", y="train_loss", color="run", markers=True, title="Training loss (Huber, scaled)")
                fig.update_traces(line=dict(width=2.5))
                show(fig, 400)
            with c2:
                fig = px.line(d, x="epoch", y="val_mae", color="run", markers=True, title="Validation MAE (veh/h)")
                fig.update_traces(line=dict(width=2.5))
                show(fig, 400)
            fig = px.line(d, x="epoch", y="lr", color="run", log_y=True, markers=True,
                          title="Learning rate (ReduceLROnPlateau)")
            fig.update_traces(line=dict(width=2.5))
            show(fig, 340)
    else:
        st.info("No training histories found.")

    if A["tuning"] is not None:
        st.subheader("Random search (validation MAE)")
        t = A["tuning"]
        fig = px.strip(t, x="model", y="val_MAE", color="model", hover_data=t.columns.tolist(),
                       title="Each point is one trial")
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        show(fig, 420)
        st.dataframe(t, use_container_width=True, hide_index=True)
    if A["best"]:
        st.subheader("Selected configuration")
        st.json(A["best"])
    if A["config"]:
        with st.expander("Experiment configuration"):
            st.json(A["config"])
