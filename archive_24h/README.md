# Hourly Traffic Volume Forecasting on I-94 with Deep Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://metro-traffic-eizdmd8ishaymreetxc3ez.streamlit.app/)

**Live dashboard:** <https://metro-traffic-eizdmd8ishaymreetxc3ez.streamlit.app/>

P4AIDS (Python for AI and Data Science, HCMUT) assignment — **Demand Forecasting** topic.

Given the last 168 hours of traffic, weather and calendar data plus the known calendar of the next 24 hours,
forecast the hourly westbound traffic volume on Interstate 94 (Minneapolis–St Paul) for the next 24 hours.
Dataset: [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (CC BY 4.0).

## Project structure

```
Metro traffic/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv.gz   raw data (auto-downloaded from UCI if missing)
├── src/metro_traffic/                           the implementation (one package, reused everywhere)
│   ├── config.py      experiment settings (dataclass)
│   ├── data.py        loading, data-quality checks, cleaning, hourly grid, gap handling
│   ├── features.py    feature engineering, train-only scaling, chronological split, sliding windows
│   ├── baselines.py   daily-naive, seasonal-naive (weekly), Ridge regression
│   ├── metrics.py     MAE, RMSE, sMAPE, R², rMAE and a results tracker
│   ├── models.py      MLP, LSTM, GRU, TCN, PatchTransformer encoders + shared forecast head
│   ├── training.py    GPU window batcher, training loop (AdamW, Huber, early stopping, AMP), prediction
│   ├── explain.py     permutation feature importance, Integrated Gradients
│   ├── pipeline.py    Experiment class: runs every step of the workflow
│   ├── artifacts.py   saves / loads the outputs used by the dashboard
│   └── report.py      result tables + static figures (used by export_results.py and the dashboard)
├── notebooks/
│   └── P4AIDS_Metro_Traffic_DL.ipynb            report notebook (problem → literature → models → evaluation)
├── train.py                                     run the whole experiment from the command line
├── export_results.py                            export figures / tables / summary into result/
├── dashboard/
│   ├── app.py                                   Streamlit dashboard
│   └── requirements.txt                         light dependencies for the online dashboard
├── .streamlit/config.toml                       dashboard theme (light, high contrast)
├── outputs/                                     raw artifacts created by train.py / the notebook
├── result/                                      exported figures (PNG), tables (CSV) and summary.md
└── requirements.txt
```

## 1. Run the experiment

### Option A — Google Colab (recommended, free GPU)
1. Upload the whole `Metro traffic` folder to Google Drive, e.g. `MyDrive/P4AIDS/Metro traffic`.
2. Open `notebooks/P4AIDS_Metro_Traffic_DL.ipynb` with Colab.
3. Check `PROJECT_DIR` in the first code cell, select **Runtime → Change runtime type → T4 GPU**, then **Run all**.
4. The notebook writes its results to `Metro traffic/outputs/` on your Drive.

### Option B — command line
```bash
pip install -r requirements.txt
pip install torch                            # only needed for training
python train.py                              # full run: 5 models + tuning + ablation + explanations
python train.py --no-tuning --epochs 20      # faster
python train.py --models LSTM GRU            # only some architectures
python train.py --baselines-only             # no PyTorch needed
```

## 2. Open the dashboard
Copy the `outputs/` folder produced on Colab into this folder (if you trained on Colab), then run from this folder
(so that `.streamlit/config.toml` is used):
```bash
pip install streamlit plotly pandas numpy matplotlib
streamlit run dashboard/app.py
```
The dashboard only reads `outputs/`, so it needs neither PyTorch nor a GPU. Pages:

| Page | Content |
|---|---|
| Overview | task, best model KPIs, leaderboard, data splits, data-quality findings, pipeline |
| Data Explorer | interactive time series, weekday × hour heatmap, day-type profiles, weather effects |
| Model Comparison | metric bars for all models, error by forecast horizon, accuracy vs. size/time |
| Forecast Explorer | any 24-hour forecast in the test period, or day-ahead forecasts stitched over several days |
| Error Analysis | error by hour / day type / month, residuals, predicted vs actual, worst days |
| Explainability | permutation importance, Integrated Gradients by lag, feature and horizon |
| Training & Tuning | learning curves, learning-rate schedule, random-search trials, selected configuration |

## 3. Save the results
```bash
python export_results.py          # reads outputs/, writes result/
```
or click **Export figures, tables & summary** in the dashboard sidebar. `result/` contains:
- `figures/` — 13 high-resolution PNG charts (leaderboard, error by horizon, forecast examples, error breakdown,
  predicted vs actual, permutation importance, Integrated Gradients, learning curves, tuning trials …)
- `tables/` — CSV tables behind the charts (leaderboard, MAE by horizon / hour / day type / month, worst days …)
- `summary.md` — key numbers, leaderboard and all figures in one markdown report

## 4. Publish the dashboard online (Streamlit Community Cloud)
1. Push this folder to a GitHub repository (`outputs/` must be included — the dashboard reads it).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
3. Repository: your repo · Branch: `main` · Main file path: `dashboard/app.py` · (Advanced settings: Python 3.11).
4. Click **Deploy**. Dependencies come from `dashboard/requirements.txt` (no PyTorch), the theme from `.streamlit/config.toml`.
5. Share the app link — this project is live at <https://metro-traffic-eizdmd8ishaymreetxc3ez.streamlit.app/>. Every `git push` redeploys the app automatically.

## Method summary
- **Cleaning:** 7,629 duplicated timestamps merged (weather labels → multi-hot); sensor errors fixed; hourly grid;
  gaps ≤ 3 h interpolated, longer gaps (incl. a 307-day outage) excluded from windows; holiday label spread from
  the 00:00 row to the whole day.
- **Split:** train ≤ 2016, validation = 2017, test = Jan–Sep 2018 (chronological, model selection on validation).
- **Baselines:** daily-naive, seasonal-naive (weekly), Ridge on 168 lags.
- **Deep learning:** five encoders sharing a head that receives the future calendar and a learned weekly-seasonal
  skip; Huber loss, AdamW, ReduceLROnPlateau, gradient clipping, dropout, early stopping, mixed precision;
  random-search tuning; ablation without weather.
- **Explanation:** permutation importance by feature group and Integrated Gradients.
