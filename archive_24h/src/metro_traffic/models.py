"""PyTorch forecasting models.

Every model = encoder(past window) -> embedding, then a shared head that also
receives the known-future calendar features and adds a learned weekly
seasonal skip (y[t+h-168]). Only the encoder differs, so the comparison is fair.

    X_past (B, L, F) --> Encoder --> z (B, D) --+
                                                +-- concat --> MLP head --> + w_h * y_season --> y_hat (B, H)
    X_future_calendar (B, H, C) --> flatten ----+
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HP = {
    "MLP": dict(hidden=512, d_emb=256, dropout=0.2),
    "LSTM": dict(hidden=128, layers=2, dropout=0.2),
    "GRU": dict(hidden=128, layers=2, dropout=0.2),
    "TCN": dict(channels=64, kernel=3, dropout=0.1),
    "PatchTransformer": dict(patch=24, d_model=128, heads=4, layers=2, dropout=0.1),
}

SEARCH_SPACES = {
    "MLP": dict(hidden=[256, 512, 1024], d_emb=[128, 256], dropout=[0.1, 0.2, 0.3]),
    "LSTM": dict(hidden=[64, 128, 192, 256], layers=[1, 2, 3], dropout=[0.1, 0.2, 0.3]),
    "GRU": dict(hidden=[64, 128, 192, 256], layers=[1, 2, 3], dropout=[0.1, 0.2, 0.3]),
    "TCN": dict(channels=[32, 64, 96, 128], kernel=[2, 3], dropout=[0.0, 0.1, 0.2]),
    "PatchTransformer": dict(patch=[12, 24], d_model=[64, 128, 192], heads=[4, 8], layers=[1, 2, 3],
                             dropout=[0.1, 0.2]),
}
COMMON_SPACE = dict(lr=[3e-4, 1e-3, 2e-3], wd=[1e-5, 1e-4, 1e-3], batch_size=[128, 256, 512],
                    head_hidden=[128, 256, 512])


class ForecastModel(nn.Module):
    def __init__(self, encoder: nn.Module, d_emb: int, horizon: int, n_fut: int,
                 head_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(d_emb + horizon * n_fut, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon))
        self.season_w = nn.Parameter(torch.full((horizon,), 0.5))

    def forward(self, x_past, x_fut, y_season):
        z = self.encoder(x_past)
        out = self.head(torch.cat([z, x_fut.flatten(1)], dim=1))
        return out + self.season_w * y_season


class MLPEncoder(nn.Module):
    def __init__(self, lookback, n_past, hidden=512, d_emb=256, dropout=0.2, **_):
        super().__init__()
        self.d_emb = d_emb
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(lookback * n_past, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_emb), nn.GELU())

    def forward(self, x):
        return self.net(x)


class RNNEncoder(nn.Module):
    def __init__(self, n_past, hidden=128, layers=2, dropout=0.2, cell="lstm", **_):
        super().__init__()
        self.d_emb = hidden
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_past, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.norm(out[:, -1])


class CausalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection (Bai et al., 2018)."""

    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        wn = nn.utils.parametrizations.weight_norm
        self.c1 = wn(nn.Conv1d(channels, channels, kernel, dilation=dilation))
        self.c2 = wn(nn.Conv1d(channels, channels, kernel, dilation=dilation))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.gelu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.gelu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.gelu(x + y)


class TCNEncoder(nn.Module):
    def __init__(self, n_past, channels=64, kernel=3, dilations=(1, 2, 4, 8, 16, 32, 64), dropout=0.1, **_):
        super().__init__()
        self.d_emb = channels
        self.inp = nn.Conv1d(n_past, channels, 1)
        self.blocks = nn.Sequential(*[CausalBlock(channels, kernel, d, dropout) for d in dilations])

    def forward(self, x):                      # x: (B, L, F)
        y = self.blocks(self.inp(x.transpose(1, 2)))
        return y[:, :, -1]                     # last time step sees the whole receptive field


class PatchTransformerEncoder(nn.Module):
    """PatchTST-style: the 168-h window is cut into patches that become Transformer tokens."""

    def __init__(self, lookback, n_past, patch=24, d_model=128, heads=4, layers=2, dropout=0.1, **_):
        super().__init__()
        if lookback % patch:
            raise ValueError("lookback must be divisible by patch")
        self.patch, self.n_patch = patch, lookback // patch
        self.proj = nn.Linear(patch * n_past, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patch, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_emb = d_model * self.n_patch

    def forward(self, x):
        B, _, F_ = x.shape
        p = x.reshape(B, self.n_patch, self.patch * F_)
        z = self.enc(self.proj(p) + self.pos)
        return self.norm(z).flatten(1)


def build_model(kind: str, n_past: int, n_fut: int, lookback: int, horizon: int, device=None, **hp) -> ForecastModel:
    hp = dict(hp)
    head_hidden = hp.pop("head_hidden", 256)
    dropout = hp.get("dropout", 0.1)
    if kind == "MLP":
        enc = MLPEncoder(lookback, n_past, **hp)
    elif kind == "LSTM":
        enc = RNNEncoder(n_past, cell="lstm", **hp)
    elif kind == "GRU":
        enc = RNNEncoder(n_past, cell="gru", **hp)
    elif kind == "TCN":
        enc = TCNEncoder(n_past, **hp)
    elif kind == "PatchTransformer":
        enc = PatchTransformerEncoder(lookback, n_past, **hp)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model = ForecastModel(enc, enc.d_emb, horizon, n_fut, head_hidden=head_hidden, dropout=dropout)
    return model.to(device) if device is not None else model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
