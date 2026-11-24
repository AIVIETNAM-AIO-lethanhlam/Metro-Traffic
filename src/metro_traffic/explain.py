"""Explanations of the selected DL model (docs/methodology.md §7, group explain.py).

* Integrated Gradients: 256 random test windows (rng seed 42), 24 integration steps
  (alpha = linspace(0, 1, 24)), zero baseline in standardised space. |IG| is averaged over the
  windows and then aggregated by feature (mean over the 168 positions) and by lag (mean over
  features), in standardised units, as in the group code.
* Weather occlusion: on the same windows, replace each weather category by the unknown token and
  measure the mean absolute change of the prediction.

The group explains the seed-42 fine-tuned checkpoint; here the explanation covers the fine-tuned
seed ensemble that produces the reported forecast (post-merge plan 4.2). IG is linear in the
model output, so the ensemble IG is the mean of the per-seed IGs; the occlusion uses the mean
prediction of the seed models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .training import WindowBatcher, predict


def ensemble_predict(models, batcher: WindowBatcher, pos, transform=None) -> np.ndarray:
    return np.mean([predict(m, batcher, pos, transform=transform) for m in models], axis=0)


def sample_test_windows(pos, n: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos)
    return pos[rng.choice(len(pos), size=min(n, len(pos)), replace=False)]


def integrated_gradients(model, batcher: WindowBatcher, pos, steps: int = 24) -> np.ndarray:
    """IG of the standardised output w.r.t. the standardised numeric window, shape (N, L, F)."""
    model.eval()
    num, cat, _ = batcher.get(pos)
    num = num.float()
    total = torch.zeros_like(num)
    # cuDNN RNNs refuse backward() in eval mode -> disable cuDNN while taking gradients
    with torch.backends.cudnn.flags(enabled=False):
        for alpha in torch.linspace(0, 1, steps, device=num.device):
            inp = (alpha * num).detach().requires_grad_(True)
            grad, = torch.autograd.grad(model(inp, cat).sum(), inp)
            total += grad
    return (num * total / steps).detach().cpu().numpy()


def ig_report(models, batcher: WindowBatcher, pos_sample, steps: int = 24, chunk: int = 64):
    """Returns (feature table, lag table, (L, F) matrix of mean |IG|) for the seed ensemble."""
    abs_sum = None
    for i in range(0, len(pos_sample), chunk):
        part = pos_sample[i:i + chunk]
        ig = np.mean([integrated_gradients(m, batcher, part, steps) for m in models], axis=0)
        s = np.abs(ig).sum(0)
        abs_sum = s if abs_sum is None else abs_sum + s
    mat = abs_sum / len(pos_sample)                                     # (L, F)
    L = mat.shape[0]
    feat = (pd.DataFrame({"feature": batcher.columns, "mean_abs_attr": mat.mean(0)})
            .sort_values("mean_abs_attr", ascending=False).reset_index(drop=True))
    lag = pd.DataFrame({"lag": np.arange(-L + 1, 1), "mean_abs_ig": mat.mean(1)})
    return feat, lag, mat


def weather_occlusion(models, batcher: WindowBatcher, pos_sample, vocab: dict) -> pd.DataFrame:
    if not batcher.use_cat:
        return pd.DataFrame()
    inv = {v: k for k, v in vocab.items()}
    _, cat, _ = batcher.get(pos_sample)
    base = ensemble_predict(models, batcher, pos_sample)
    rows = []
    for code in torch.unique(cat).tolist():
        if code == batcher.unk:
            continue

        def tf(x, c, code=code):
            return x, torch.where(c == code, torch.full_like(c, batcher.unk), c)

        p = ensemble_predict(models, batcher, pos_sample, transform=tf)
        delta = float(np.mean(np.abs(p - base)))
        rows.append(dict(weather_code=int(code), weather=inv.get(code, "Unknown"),
                         windows_with_category=int((cat == code).any(dim=1).sum().item()),
                         mean_abs_delta=delta, mean_abs_delta_scaled=delta / batcher.y_std))
    return pd.DataFrame(rows).sort_values("mean_abs_delta", ascending=False).reset_index(drop=True)
