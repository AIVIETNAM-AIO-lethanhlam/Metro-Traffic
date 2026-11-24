# One-hour-ahead Traffic Volume Forecasting on I-94 — Deep Learning part

P4AIDS (Python for AI and Data Science, HCMUT) group assignment — **Demand Forecasting** topic.
This folder is the **deep learning** part. It follows the group's problem definition and data contract
(`p4aids-quickeda`: `docs/problem_definition.md`, `docs/methodology.md`), so the DL numbers can be compared
directly with the ML models of the other members.

**Task.** At each hour *t*, forecast the westbound traffic volume of that hour on Interstate 94 (Minneapolis–St Paul)
one hour ahead. The DL models read a 168-hour window of the shared feature rows: traffic history up to *t − 1*
(lags, rolling statistics, weekly profile), the calendar of *t* and the weather observed at *t*.
Dataset: [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (CC BY 4.0).

**Shared with the ML part:** data from 2016-01-01, chronological 70 / 15 / 15 split by time span
(14,016 hours = 9,306 / 2,054 / 2,656), 46 numeric features + `weather_main`, MAE / RMSE / R² / sMAPE / MASE,
Diebold–Mariano tests. Everything is chosen on validation; the test set is used once.

## Project structure

```
Metro traffic/
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv.gz   raw data (auto-downloaded from UCI if missing)
│   └── processed/                               group processed files (parquet / csv.gz), rebuilt if missing
├── src/metro_traffic/
│   ├── config.py      data-contract constants + experiment settings (seeds, epochs, tuning, ablation)
│   ├── data.py        cleaning, hourly grid, features, split — port of the group's data.py
│   ├── windows.py     168-h windows, train-only imputation/scaling, weather_main codes
│   ├── baselines.py   persistence, seasonal naive (t-24), hour-of-week mean
│   ├── metrics.py     MAE, RMSE, R², sMAPE, MASE, Diebold–Mariano (Newey–West, 24 lags)
│   ├── models.py      LSTMRegressor and TCNRegressor (same layers as the group dl.py) + weather embedding
│   ├── training.py    GPU window batcher, group training protocol, fine-tuning, prediction
│   ├── tuning.py      Optuna TPE, group search space (random-search fallback)
│   ├── explain.py     Integrated Gradients and weather occlusion (seed ensemble)
│   ├── pipeline.py    Experiment class: the whole workflow with several seeds
│   ├── artifacts.py   saves / loads the outputs used by the dashboard
│   └── report.py      tables + static figures (export_results.py and the dashboard)
├── notebooks/P4AIDS_Metro_Traffic_DL.ipynb      report notebook (problem → literature → models → evaluation)
├── train.py                                     the same experiment from the command line
├── export_results.py                            figures / tables / summary -> result/
├── dashboard/app.py                             Streamlit dashboard (reads outputs/ only)
├── outputs/                                     artifacts written by train.py / the notebook
└── result/                                      exported figures (PNG), tables (CSV), summary.md
```

## Workflow (group `docs/methodology.md` §5–7)
1. Shared processed data → 168-h windows of the 46 columns + `weather_main` codes. Missing lag cells inside a
   window (1.1 %) = training median; channels standardised with training-window statistics.
2. Baselines: persistence (t−1), seasonal naive (t−24, also the MASE scale), hour-of-week mean.
3. **Two architectures, as in the group pipeline:**
   - **LSTM:** embedding + 1–2 LSTM layers (32–128 hidden), last hidden state → MLP head;
   - **TCN:** embedding + 1×1 input conv + 4–5 residual dilated blocks (dilations 1…16, kernel 3) →
     [last step, mean over time] → MLP head.
4. **Optuna tuning of both** (10 trials each, TPE seed 42, ≤ 25 epochs, patience 5): hidden, depth, dropout,
   lr, weight decay, batch size, embedding size (4 / 8).
5. **Final training with seeds 42, 43, 44:** SmoothL1 (β 0.5), AdamW, ReduceLROnPlateau (0.5 / 2), gradient
   clipping 1.0, early stopping on validation MAE (patience 10, ≤ 80 epochs) → ensemble = mean of the seeds.
6. **Fine-tuning:** each seed model 8 more epochs on train + validation at lr/5 with cosine decay; test is
   reported before and after fine-tuning.
7. **Selection:** best DL model = lowest validation MAE of the seed ensemble before fine-tuning.
8. **Explanation:** Integrated Gradients (256 random test windows, 24 steps, zero baseline in standardised
   space, by feature and by lag) and weather occlusion (mean |change of the prediction|).
9. **Diebold–Mariano** (absolute errors, Newey–West 24 lags): best ML (by validation MAE, from the group's
   `ml_predictions.parquet`) vs best DL; also best DL vs LSTM/TCN and the baselines.
10. Extra (post-merge plan 4.3): cumulative feature-group ablation of the best DL model
    (calendar → + weather → + lags → + engineered), all seeds.

Differences from the group `dl.py` that do not change the method: windows are cut on the GPU from the grid
(no materialised N × 168 × 46 array, much faster); training-window statistics are computed exactly in float64;
the seed is set before the model is built; IG and occlusion use the fine-tuned seed ensemble instead of only the
seed-42 checkpoint (post-merge plan 4.2).

**Group reference run** (`docs/methodology.md` §9): TCN val 161.6 / test 126.6, LSTM val 173.5 / test 131.2,
best ML `lightgbm_blend` val 152.5 / test 119.7.

## 1. Run the experiment

### Option A — Google Colab (recommended, free GPU)
1. Upload the whole `Metro traffic` folder to Google Drive, e.g. `MyDrive/P4AIDS/Metro traffic`.
2. Open `notebooks/P4AIDS_Metro_Traffic_DL.ipynb` with Colab, choose **Runtime → Change runtime type → T4 GPU**.
3. Check `PROJECT_DIR` in the first code cell, then **Run all** (roughly 1–2 h on a T4 for 2 models ×
   (10 trials + 3 seeds) + ablation). Results go to `outputs/` and `result/` on your Drive.

### Option B — command line
```bash
pip install -r requirements.txt
python train.py --smoke                       # 2-minute check that everything runs (writes outputs_smoke/)
python train.py                               # full run (LSTM + TCN)
python train.py --seeds 42 43 44 45 46        # more seeds per ensemble
python train.py --models TCN --no-tuning --no-ablation
python train.py --ml-results ml_results      # folder with the group's results/ml_predictions.parquet
python train.py --drop-leaky                  # without how_mean / lag1_minus_how
python train.py --baselines-only              # no PyTorch needed
```
**ML predictions for the Diebold–Mariano test:** copy `results/ml_predictions.parquet` from the group
repository into `ml_results/` (read-only use of the group files; nothing is written there).

To reuse the group's processed files, copy `metro_traffic_model_frame.parquet`, `metro_traffic_full_grid.parquet`
and `metro_traffic_meta.json` into `data/processed/` (or pass `--processed <folder>`); otherwise they are rebuilt
with the same rules.

## 2. Open the dashboard
Copy the `outputs/` folder produced on Colab into this folder, then run from this folder:
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

| Page | Content |
|---|---|
| Overview | task, KPIs of the selected model, test leaderboard, splits, pipeline |
| Data Explorer | daily series by split, weekday × hour heat map, day-type profiles, weather |
| Model Comparison | any metric on test / validation, Diebold–Mariano tests |
| Seeds & Fine-tuning | single seeds vs ensemble for every encoder, effect of fine-tuning |
| Forecast Explorer | one-hour-ahead forecasts on any test window |
| Error Analysis | MAE by hour / weekday / day type / month / weather, forecast vs actual, largest errors |
| Ablation | cumulative feature groups, mean ± std over seeds |
| Explainability | Integrated Gradients by feature and lag, IG heat map, weather occlusion |
| Training & Tuning | Optuna trials, learning curves of every seed, fine-tuning loss |

## 3. Save the results
```bash
python export_results.py          # reads outputs/, writes result/
```
or click **Export figures, tables & summary** in the dashboard sidebar: 10 PNG figures, CSV tables and `summary.md`.

## 4. Publish the dashboard online (Streamlit Community Cloud)
1. Push this folder to GitHub (`outputs/` must be included — the dashboard reads it).
2. <https://share.streamlit.io> → **Create app** → repo, branch `main`, main file `dashboard/app.py`.
3. Dependencies come from `dashboard/requirements.txt` (no PyTorch), the theme from `.streamlit/config.toml`.
4. Live app: <https://metro-traffic-eizdmd8ishaymreetxc3ez.streamlit.app/> — every `git push` redeploys it.

## Previous version
The earlier 24-hour-ahead version (own split and features) is kept in `archive_24h/` and in git commit `a0ddba2`.
