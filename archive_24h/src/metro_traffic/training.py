"""Window batching on the GPU, training loop with early stopping, and prediction."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .features import SEASON_LAG, Dataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WindowBatcher:
    """Cuts (past, future-calendar, target, seasonal-naive) windows straight from the full
    series stored on the device, so the N x 168 x F tensor is never materialised."""

    def __init__(self, ds: Dataset, device: torch.device, past_cols: list[str] | None = None):
        cols = past_cols or ds.past_cols
        keep = [ds.past_cols.index(c) for c in cols]
        self.past_cols = cols
        self.fut_cols = ds.fut_cols
        self.device = device
        self.Xp = torch.tensor(ds.X_past[:, keep], device=device)
        self.Xf = torch.tensor(ds.X_fut, device=device)
        self.y = torch.tensor(ds.y_sc, device=device)
        self.po = torch.tensor(ds.past_off, device=device)
        self.fo = torch.tensor(ds.fut_off, device=device)
        self.y_mean, self.y_std = ds.y_mean, ds.y_std
        self.lookback, self.horizon = ds.lookback, ds.horizon

    @property
    def n_past(self) -> int:
        return self.Xp.shape[1]

    @property
    def n_fut(self) -> int:
        return self.Xf.shape[1]

    def get(self, origins):
        o = torch.as_tensor(np.asarray(origins), device=self.device)
        pi = o[:, None] + self.po
        fi = o[:, None] + self.fo
        return self.Xp[pi], self.Xf[fi], self.y[fi], self.y[fi - SEASON_LAG]

    def iterate(self, origins, batch_size: int, shuffle: bool = False):
        origins = np.asarray(origins)
        order = np.random.permutation(len(origins)) if shuffle else np.arange(len(origins))
        for i in range(0, len(origins), batch_size):
            yield self.get(origins[order[i:i + batch_size]])


def _amp(device) -> bool:
    return device.type == "cuda"


@torch.no_grad()
def predict(model: nn.Module, batcher: WindowBatcher, origins, batch_size: int = 1024, transform=None) -> np.ndarray:
    """Forecasts in vehicles/hour, shape (N, H). `transform(xp, xf)` lets callers perturb inputs."""
    model.eval()
    outs = []
    for xp, xf, _, ys in batcher.iterate(origins, batch_size):
        if transform is not None:
            xp, xf = transform(xp, xf)
        with torch.autocast(device_type=batcher.device.type, dtype=torch.float16, enabled=_amp(batcher.device)):
            outs.append(model(xp, xf, ys).float().cpu().numpy())
    return np.concatenate(outs) * batcher.y_std + batcher.y_mean


def train_model(model: nn.Module, batcher: WindowBatcher, idx_train, idx_val, y_val: np.ndarray,
                epochs: int = 40, lr: float = 1e-3, wd: float = 1e-4, batch_size: int = 256,
                patience: int = 6, verbose: bool = True):
    """AdamW + Huber loss + ReduceLROnPlateau + gradient clipping + AMP + early stopping.

    Returns (model with best weights, history DataFrame, best val MAE, seconds).
    """
    device = batcher.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp(device))
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best, best_state, wait, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xp, xf, y, ys in batcher.iterate(idx_train, batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_amp(device)):
                loss = loss_fn(model(xp, xf, ys).float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)

        val_mae = float(np.mean(np.abs(np.clip(predict(model, batcher, idx_val), 0, None) - y_val)))
        sched.step(val_mae)
        lr_now = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=ep, train_loss=total / n, val_mae=val_mae, lr=lr_now))
        improved = val_mae < best - 1e-3
        if improved:
            best, wait = val_mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if verbose:
            print(f"  epoch {ep:02d} | train_loss {total / n:.4f} | val_MAE {val_mae:7.1f} | lr {lr_now:.1e}"
                  + ("  *" if improved else ""))
        if wait >= patience:
            if verbose:
                print(f"  early stopping at epoch {ep}")
            break
    if best_state is not None:        # None only if validation MAE was NaN in every epoch
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best, time.time() - t0
