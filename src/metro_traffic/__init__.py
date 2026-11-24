"""Metro Interstate Traffic Volume - deep learning part of the P4AIDS group project.

Task (shared with the ML members): forecast traffic_volume(t) one hour ahead from a 168-hour
window of the shared feature columns (traffic history up to t-1, calendar of t, weather
observed at t), with the group's 70/15/15 chronological split from 2016-01-01.

Modules
-------
config      : data contract constants + experiment settings
data        : raw -> cleaned hourly grid -> features -> split (same rules as the group pipeline)
windows     : 168-h windows, train-only imputation and scaling, weather_main codes (NumPy)
baselines   : persistence, seasonal naive, hour-of-week mean
metrics     : MAE / RMSE / R2 / sMAPE / MASE and the Diebold-Mariano test
models      : LSTM and TCN regressors of the group pipeline              (PyTorch)
training    : GPU window batcher, training protocol, fine-tuning, prediction (PyTorch)
tuning      : Optuna (or random-search fallback) on validation MAE         (PyTorch)
explain     : Integrated Gradients and weather occlusion                (PyTorch)
pipeline    : Experiment class running the whole workflow with several seeds (PyTorch)
artifacts   : save / load the outputs read by the dashboard
report      : tables and static figures for export_results.py and the dashboard

Only `models`, `training`, `tuning`, `explain` and `pipeline` need PyTorch.
"""

__version__ = "2.0.0"
