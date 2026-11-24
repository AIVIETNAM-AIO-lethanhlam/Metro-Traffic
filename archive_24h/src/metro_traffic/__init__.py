"""Metro Interstate Traffic Volume — 24-hour deep learning forecasting.

Modules
-------
config      : experiment configuration (dataclass)
data        : loading + cleaning the raw UCI file
features    : feature engineering, scaling, chronological split, sliding windows
baselines   : naive and Ridge baselines (no deep learning)
metrics     : MAE / RMSE / sMAPE / R2 / rMAE + a results tracker
models      : PyTorch encoders (MLP, LSTM, GRU, TCN, PatchTransformer) and forecast head
training    : GPU window batcher, training loop with early stopping, prediction
explain     : permutation importance and Integrated Gradients
pipeline    : Experiment class that runs the full workflow step by step
artifacts   : save / load the outputs consumed by the Streamlit dashboard

Modules that need PyTorch (models, training, explain) are imported lazily, so
the data, baseline and artifact code (and the dashboard) work without torch.
"""

__version__ = "1.0.0"
