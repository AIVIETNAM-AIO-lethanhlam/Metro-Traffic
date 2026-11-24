"""The two DL architectures of the group pipeline (docs/methodology.md §5, group dl.py).

* LSTM: weather embedding + 1-2 LSTM layers, last hidden state -> MLP head.
* TCN : weather embedding + 1x1 input conv + 4 (layers = 1) or 5 (layers = 2) residual dilated
        blocks (dilations 1, 2, 4, 8 (, 16), kernel 3) -> [last step, mean over time] -> MLP head.

Both read a (B, 168, F) numeric window and the (B, 168) `weather_main` codes and return the
standardised one-hour-ahead forecast (B,). Same layers as the group code; the only addition is
`use_cat=False` (no embedding), used by the ablation stage without weather.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_NAMES = ("LSTM", "TCN")

# group defaults when tuning is switched off (dl.py `run`, tune=False)
DEFAULT_CFG = dict(hidden=64, layers=2, dropout=0.1, lr=1e-3, wd=1e-4, batch=256, emb=8)

# group Optuna space, shared by both architectures (dl.py `suggest`)
SEARCH_SPACE = dict(
    hidden=[32, 64, 96, 128],          # categorical
    layers=(1, 2, "int"),
    dropout=(0.0, 0.35),
    lr=(1e-4, 3e-3, "log"),
    wd=(1e-6, 1e-3, "log"),
    batch=[128, 256, 512],
    emb=[4, 8],
)


class LSTMRegressor(nn.Module):
    def __init__(self, n_num: int, n_cat: int, emb: int, hidden: int, layers: int, dropout: float,
                 use_cat: bool = True):
        super().__init__()
        self.emb = nn.Embedding(n_cat, emb) if use_cat else None
        n_in = n_num + (emb if use_cat else 0)
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, num: torch.Tensor, cat: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([num, self.emb(cat)], dim=-1) if self.emb is not None else num
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, kernel, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(),
            nn.Dropout(dropout))
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.down(x))


class TCNRegressor(nn.Module):
    def __init__(self, n_num: int, n_cat: int, emb: int, hidden: int, layers: int, dropout: float,
                 use_cat: bool = True):
        super().__init__()
        self.emb = nn.Embedding(n_cat, emb) if use_cat else None
        n_in = n_num + (emb if use_cat else 0)
        self.inp = nn.Conv1d(n_in, hidden, 1)
        depth = 4 if layers == 1 else 5            # dilations 1,2,4,8 (,16): receptive field 61 / 125 h
        self.blocks = nn.Sequential(*[TemporalBlock(hidden, hidden, 3, dilation=2 ** i, dropout=dropout)
                                      for i in range(depth)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, num: torch.Tensor, cat: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([num, self.emb(cat)], dim=-1) if self.emb is not None else num
        x = self.blocks(self.inp(x.transpose(1, 2)))             # (B, C, L)
        feat = torch.cat([x[:, :, -1], x.mean(dim=-1)], dim=-1)
        return self.head(feat).squeeze(-1)


def build_model(name: str, cfg: dict, n_num: int, n_cat: int, use_cat: bool = True, device=None) -> nn.Module:
    kw = dict(n_num=n_num, n_cat=max(n_cat, 1), emb=int(cfg["emb"]), hidden=int(cfg["hidden"]),
              layers=int(cfg["layers"]), dropout=float(cfg["dropout"]), use_cat=use_cat)
    if name == "LSTM":
        model = LSTMRegressor(**kw)
    elif name == "TCN":
        model = TCNRegressor(**kw)
    else:
        raise ValueError(f"unknown model: {name} (expected one of {MODEL_NAMES})")
    return model.to(device) if device is not None else model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
