"""Model explanation: permutation feature importance and Integrated Gradients."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .features import Dataset
from .training import WindowBatcher, predict

CYCLIC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
OFFDAY = ["is_weekend", "is_holiday", "is_offday"]


def feature_groups(ds: Dataset) -> dict[str, tuple[list[str], list[str]]]:
    """group name -> (past columns, future columns)."""
    return {
        "Traffic history": (["traffic_scaled"], []),
        "Temperature": (["temp_c"], []),
        "Rain / snow": (["rain_log", "snow_log"], []),
        "Clouds + weather type": (["clouds"] + ds.wx_cols, []),
        "Past calendar (hour/day/season)": (CYCLIC, []),
        "Future calendar (hour/day/season)": ([], CYCLIC),
        "Weekend / holiday (past)": (OFFDAY, []),
        "Weekend / holiday (future)": ([], OFFDAY),
    }


def permutation_importance(model, batcher: WindowBatcher, ds: Dataset, origins, n_repeats: int = 3) -> pd.DataFrame:
    """Shuffle one feature group across test windows and measure how much MAE increases."""
    y_true = ds.targets(origins)
    base_mae = float(np.mean(np.abs(np.clip(predict(model, batcher, origins), 0, None) - y_true)))
    rows = []
    for group, (pcols, fcols) in feature_groups(ds).items():
        pi = [batcher.past_cols.index(c) for c in pcols if c in batcher.past_cols]
        fi = [batcher.fut_cols.index(c) for c in fcols]
        if not pi and not fi:
            continue
        deltas = []
        for rep in range(n_repeats):
            gen = torch.Generator(device=batcher.device)
            gen.manual_seed(rep)

            def tf(xp, xf, pi=pi, fi=fi, gen=gen):
                perm = torch.randperm(xp.shape[0], device=xp.device, generator=gen)
                xp, xf = xp.clone(), xf.clone()
                if pi:
                    xp[:, :, pi] = xp[perm][:, :, pi]
                if fi:
                    xf[:, :, fi] = xf[perm][:, :, fi]
                return xp, xf

            p = np.clip(predict(model, batcher, origins, transform=tf), 0, None)
            deltas.append(float(np.mean(np.abs(p - y_true))) - base_mae)
        rows.append(dict(group=group, delta_MAE=float(np.mean(deltas)), std=float(np.std(deltas)),
                         base_MAE=base_mae))
    return pd.DataFrame(rows).sort_values("delta_MAE", ascending=False).reset_index(drop=True)


def integrated_gradients(model, batcher: WindowBatcher, origins, steps: int = 32, target_h=None) -> np.ndarray:
    """IG attributions of the past window, shape (N, L, F). Baseline = 0 (the training mean
    after standardisation). target_h=None explains the sum of all 24 outputs."""
    model.eval()
    xp, xf, _, ys = batcher.get(origins)
    xp = xp.float()
    base = torch.zeros_like(xp)
    total = torch.zeros_like(xp)
    # cuDNN RNNs refuse backward() in eval mode -> disable cuDNN while computing gradients
    with torch.backends.cudnn.flags(enabled=False):
        for a in torch.linspace(0, 1, steps, device=batcher.device):
            x = (base + a * (xp - base)).requires_grad_(True)
            out = model(x, xf, ys)
            out = out.sum() if target_h is None else out[:, target_h].sum()
            g, = torch.autograd.grad(out, x)
            total += g
    return ((xp - base) * total / steps).detach().cpu().numpy()


def ig_summary(model, batcher: WindowBatcher, ds: Dataset, origins, n_samples: int = 512,
               seed: int = 42, chunk: int = 128):
    """Returns (lag importance, feature importance, traffic-lag x horizon heatmap) DataFrames."""
    rng = np.random.default_rng(seed)
    origins = np.asarray(origins)
    sub = np.sort(rng.choice(origins, size=min(n_samples, len(origins)), replace=False))
    ig = np.concatenate([integrated_gradients(model, batcher, sub[i:i + chunk]) for i in range(0, len(sub), chunk)])
    abs_ig = np.abs(ig).mean(0)                                    # (L, F)
    L = abs_ig.shape[0]
    lag_df = pd.DataFrame({"hours_before_origin": np.arange(L - 1, -1, -1), "importance": abs_ig.sum(1)})
    feat_df = (pd.DataFrame({"feature": batcher.past_cols, "importance": abs_ig.sum(0)})
               .sort_values("importance", ascending=False).reset_index(drop=True))

    t_idx = batcher.past_cols.index("traffic_scaled") if "traffic_scaled" in batcher.past_cols else 0
    heat_sub = sub[:256]
    rows = {}
    for h in range(0, ds.horizon, 3):
        a = np.concatenate([integrated_gradients(model, batcher, heat_sub[i:i + chunk], steps=16, target_h=h)
                            for i in range(0, len(heat_sub), chunk)])
        rows[f"t+{h + 1}"] = np.abs(a[:, :, t_idx]).mean(0)
    heat_df = pd.DataFrame(rows, index=np.arange(L - 1, -1, -1)).T
    heat_df.columns.name = "hours_before_origin"
    return lag_df, feat_df, heat_df
