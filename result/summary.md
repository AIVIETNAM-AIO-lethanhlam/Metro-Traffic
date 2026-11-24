# Results summary - Metro traffic, one-hour-ahead DL (2026-11-24 23:30)

* Selected encoder (lowest validation MAE of the base seed ensemble): **TCN** (Optuna-tuned)
* Seeds per ensemble: 42, 43, 44; fine-tuned on train + val for 8 epochs
* Test: MAE 129.6, RMSE 190.1, R2 0.990, sMAPE 5.79 %, MASE 0.238
* Strongest naive reference (Hour-of-week mean): test MAE 219.2 -> 40.9 % lower MAE

## Test leaderboard

| model                 | family   | variant                             |     MAE |    RMSE |    R2 |   sMAPE |   MASE |
|:----------------------|:---------|:------------------------------------|--------:|--------:|------:|--------:|-------:|
| ML: lightgbm_blend    | ML       | nan                                 | 119.675 | 183.796 | 0.991 |   4.829 |  0.22  |
| ML: random_forest     | ML       | nan                                 | 121.548 | 189.001 | 0.99  |   4.809 |  0.223 |
| ML: lightgbm          | ML       | nan                                 | 123.202 | 188.608 | 0.99  |   4.981 |  0.226 |
| ML: xgboost           | ML       | nan                                 | 126.75  | 193.088 | 0.99  |   5.099 |  0.233 |
| TCN                   | DL       | ensemble, fine-tuned on train + val | 129.604 | 190.117 | 0.99  |   5.787 |  0.238 |
| TCN (no fine-tune)    | DL       | ensemble, train only                | 130.167 | 191.724 | 0.99  |   5.887 |  0.239 |
| LSTM                  | DL       | ensemble, fine-tuned on train + val | 133.069 | 195.46  | 0.99  |   6.182 |  0.244 |
| LSTM (no fine-tune)   | DL       | ensemble, train only                | 134.199 | 196.671 | 0.99  |   6.241 |  0.246 |
| ML: ridge             | ML       | nan                                 | 144.447 | 236.809 | 0.985 |   6.444 |  0.265 |
| Hour-of-week mean     | Baseline | nan                                 | 219.23  | 418.237 | 0.953 |   8.042 |  0.402 |
| Seasonal naive (t-24) | Baseline | nan                                 | 513.752 | 981.834 | 0.741 |  19.487 |  0.943 |
| Persistence (t-1)     | Baseline | nan                                 | 575.326 | 798.771 | 0.829 |  25.396 |  1.056 |

## Seeds (single runs vs ensemble)

| model   |   seeds |   val_MAE_mean |   val_MAE_std |   val_MAE_ensemble |   test_MAE_mean |   test_MAE_std |   test_MAE_ensemble |   test_MAE_ensemble_no_finetune |   n_params |   epochs_mean |   seconds_total |
|:--------|--------:|---------------:|--------------:|-------------------:|----------------:|---------------:|--------------------:|--------------------------------:|-----------:|--------------:|----------------:|
| TCN     |       3 |         169.18 |          3.38 |             160.23 |          136.58 |           1.53 |              129.6  |                          130.17 |     301005 |         21.33 |          137.5  |
| LSTM    |       3 |         176.75 |          1.27 |             172.39 |          136.9  |           1.12 |              133.07 |                          134.2  |     102617 |         39.33 |           84.42 |

## Ablation (mean over seeds)

|   stage | features            |   n_numeric | weather_main   |   seeds |   val_MAE |   val_std |   test_MAE |   test_std |   val_gain_vs_previous |
|--------:|:--------------------|------------:|:---------------|--------:|----------:|----------:|-----------:|-----------:|-----------------------:|
|       1 | calendar            |          14 | False          |       3 |    274.42 |      0.8  |     195.65 |       1.62 |                 nan    |
|       2 | + weather           |          20 | True           |       3 |    277.06 |      4.2  |     233.72 |       5.29 |                  -2.64 |
|       3 | + lags              |          37 | True           |       3 |    174.29 |      2.52 |     144.4  |       2.79 |                 102.77 |
|       4 | + engineered (full) |          46 | True           |       3 |    169.18 |      3.38 |     140.71 |       2.31 |                   5.12 |

## Diebold-Mariano tests (absolute errors, Newey-West 24 lags)

| model_a   | model_b               | primary   |    mean_d |   dm_stat |   p_value | better                    |
|:----------|:----------------------|:----------|----------:|----------:|----------:|:--------------------------|
| TCN       | ML: lightgbm_blend    | True      |    9.9293 |    5.7926 |    0      | B                         |
| TCN       | Persistence (t-1)     | False     | -445.722  |  -30.9925 |    0      | A                         |
| TCN       | Seasonal naive (t-24) | False     | -384.147  |  -10.2242 |    0      | A                         |
| TCN       | Hour-of-week mean     | False     |  -89.6258 |   -5.1532 |    0      | A                         |
| TCN       | LSTM (no fine-tune)   | False     |   -4.5951 |   -2.4949 |    0.0126 | A                         |
| TCN       | LSTM                  | False     |   -3.4646 |   -1.9873 |    0.0469 | A                         |
| TCN       | TCN (no fine-tune)    | False     |   -0.5634 |   -0.4814 |    0.6302 | no significant difference |
| TCN       | ML: ridge             | False     |  -14.843  |   -2.9094 |    0.0036 | A                         |
| TCN       | ML: random_forest     | False     |    8.0565 |    4.425  |    0      | B                         |
| TCN       | ML: lightgbm          | False     |    6.4019 |    3.493  |    0.0005 | B                         |
| TCN       | ML: xgboost           | False     |    2.854  |    1.5065 |    0.1319 | no significant difference |

## Files

* `tables/leaderboard_test.csv`
* `tables/leaderboard_val.csv`
* `tables/seed_summary.csv`
* `tables/seed_runs.csv`
* `tables/ablation_summary.csv`
* `tables/mae_by_hour.csv`
* `tables/mae_by_dow.csv`
* `tables/mae_by_day_type.csv`
* `tables/mae_by_month.csv`
* `tables/mae_by_weather.csv`
* `tables/worst_days_best_model.csv`
* `tables/worst_hours_best_model.csv`
* `tables/tuning_log.csv`
* `tables/diebold_mariano.csv`
* `tables/ig_feature_importance.csv`
* `tables/weather_occlusion.csv`
* `tables/data_splits.csv`
* `figures/01_leaderboard_test.png`
* `figures/02_seeds_and_ensembles.png`
* `figures/03_finetune_effect.png`
* `figures/04_forecasts_two_weeks.png`
* `figures/05_error_breakdown.png`
* `figures/06_residuals.png`
* `figures/07_ablation.png`
* `figures/08_integrated_gradients.png`
* `figures/09_weather_occlusion.png`
* `figures/10_learning_curves.png`