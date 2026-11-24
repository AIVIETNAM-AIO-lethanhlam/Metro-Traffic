"""Window batching on the device, the group training protocol, fine-tuning and prediction.

Same protocol as the group dl.py (docs/methodology.md §5):
SmoothL1 (beta 0.5) on the standardised target, AdamW(lr, wd), ReduceLROnPlateau(factor 0.5,
patience 2) on validation MAE, gradient clipping at 1.0, early stopping on validation MAE
(vehicles/hour, min improvement 1e-4) with the best epoch restored. Fine-tuning: 8 further epochs
on train + validation at lr / 5 with cosine decay (stepped once per epoch).

Implementation differences that do not change the method: windows are cut from the grid kept on
the device instead of a DataLoader over materialised windows (much faster), and the seed is set
before the model is built so that the initial weights also depend on the seed.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .windows import WindowData


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WindowBatcher:
    """Keeps the (T, F) grid on the device and cuts 168-h windows by index."""

    def __init__(self, wd: WindowData, device: torch.device, columns: list[str] | None = None,
                 use_cat: bool | None = None):
        cols = columns or wd.numeric
        keep = [wd.numeric.index(c) for c in cols]
        self.columns = list(cols)
        self.use_cat = wd.use_cat if use_cat is None else (use_cat and wd.use_cat)
        self.n_cat = wd.n_cat if self.use_cat else 0
        self.unk = wd.unk
        self.device = device
        self.X = torch.tensor(wd.X[:, keep], device=device)
        self.C = torch.tensor(wd.C, device=device)
        self.y = torch.tensor(wd.y_grid, device=device)
        self.off = torch.arange(-wd.lookback + 1, 1, device=device)
        self.y_mean, self.y_std = wd.y_mean, wd.y_std
        self.lookback = wd.lookback

    @property
    def n_num(self) -> int:
        return self.X.shape[1]

    def get(self, pos):
        p = torch.as_tensor(np.asarray(pos), device=self.device)
        idx = p[:, None] + self.off
        return self.X[idx], (self.C[idx] if self.use_cat else None), self.y[p]

    def iterate(self, pos, batch_size: int, shuffle: bool = False, rng: np.random.Generator | None = None):
        pos = np.asarray(pos)
        order = (rng or np.random.default_rng()).permutation(len(pos)) if shuffle else np.arange(len(pos))
        for i in range(0, len(pos), batch_size):
            yield self.get(pos[order[i:i + batch_size]])


@torch.no_grad()
def predict(model: nn.Module, batcher: WindowBatcher, pos, batch_size: int = 1024, transform=None) -> np.ndarray:
    """Forecasts in vehicles/hour (not clipped, as in the group code)."""
    model.eval()
    outs = []
    for x, c, _ in batcher.iterate(pos, batch_size):
        if transform is not None:
            x, c = transform(x, c)
        outs.append(model(x, c).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float64) * batcher.y_std + batcher.y_mean


def _epoch(model, batcher, pos, opt, loss_fn, batch_size, rng):
    model.train()
    total, n = 0.0, 0
    for x, c, y in batcher.iterate(pos, batch_size, shuffle=True, rng=rng):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, c), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.detach()) * len(y)
        n += len(y)
    return total / max(n, 1)


def train_model(model: nn.Module, batcher: WindowBatcher, pos_train, pos_val, y_val: np.ndarray, cfg: dict,
                max_epochs: int = 80, patience: int = 10, verbose: bool = False):
    """Returns (model with the best-epoch weights, history DataFrame, best val MAE, seconds)."""
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best_state, best, bad, hist = None, math.inf, 0, []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        tr = _epoch(model, batcher, pos_train, opt, loss_fn, int(cfg["batch"]), rng)
        val_mae = float(np.mean(np.abs(predict(model, batcher, pos_val) - y_val)))
        if not np.isfinite(val_mae):
            val_mae = math.inf
        sched.step(val_mae)
        hist.append(dict(epoch=epoch, train_loss=tr, val_mae=val_mae, lr=opt.param_groups[0]["lr"]))
        if val_mae < best - 1e-4:
            best, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose:
            print(f"    epoch {epoch:02d} | loss {tr:.4f} | val MAE {val_mae:7.1f} | lr {hist[-1]['lr']:.1e}")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best, time.time() - t0


def fine_tune(model: nn.Module, batcher: WindowBatcher, pos_trainval, cfg: dict, epochs: int = 8):
    """Group fine-tuning stage on a copy of the model; returns (fine-tuned copy, epoch losses)."""
    model = copy.deepcopy(model)
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] / 5, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    losses = []
    for _ in range(epochs):
        losses.append(_epoch(model, batcher, pos_trainval, opt, loss_fn, int(cfg["batch"]), rng))
        sched.step()
    return model, losses
