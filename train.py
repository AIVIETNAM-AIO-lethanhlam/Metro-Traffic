"""Run the full DL experiment from the command line and write the dashboard artifacts.

Examples
    python train.py                              # full run: LSTM + TCN, Optuna, 3 seeds, fine-tuning, ablation (GPU)
    python train.py --smoke                      # few-minute check that everything runs end to end
    python train.py --seeds 42 43 44 45 46       # more seeds in every ensemble
    python train.py --models TCN --no-tuning --no-ablation
    python train.py --ml-results ml_results      # folder with the group's ml_predictions.parquet (DM best ML vs best DL)
    python train.py --baselines-only             # no PyTorch needed
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metro_traffic.config import ALL_MODELS, Config  # noqa: E402


def main():
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=os.path.join(ROOT, d.raw_path))
    p.add_argument("--processed", default=os.path.join(ROOT, d.processed_dir),
                   help="folder with the group's processed files (rebuilt here if missing)")
    p.add_argument("--out", default=os.path.join(ROOT, d.out_dir))
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(d.seeds))
    p.add_argument("--epochs", type=int, default=d.max_epochs)
    p.add_argument("--trials", type=int, default=d.n_trials)
    p.add_argument("--finetune-epochs", type=int, default=d.finetune_epochs)
    p.add_argument("--ml-results", default=os.path.join(ROOT, "ml_results"),
                   help="folder with the group's ml_predictions.parquet (optional)")
    p.add_argument("--drop-leaky", action="store_true", help="drop how_mean / lag1_minus_how")
    p.add_argument("--no-tuning", action="store_true")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny budget, only to check the code runs")
    p.add_argument("--baselines-only", action="store_true")
    a = p.parse_args()

    cfg = Config(raw_path=a.raw, processed_dir=a.processed, out_dir=a.out, models=tuple(a.models),
                 seeds=tuple(a.seeds), max_epochs=a.epochs, n_trials=a.trials,
                 finetune_epochs=a.finetune_epochs, ml_results_dir=a.ml_results, drop_leaky_features=a.drop_leaky,
                 run_tuning=not a.no_tuning, run_ablation=not a.no_ablation, ablation_seeds=tuple(a.seeds))
    if a.smoke:
        cfg.max_epochs, cfg.patience, cfg.n_trials, cfg.tune_max_epochs, cfg.tune_patience = 2, 2, 2, 1, 1
        cfg.finetune_epochs, cfg.seeds, cfg.ablation_seeds, cfg.ig_samples, cfg.ig_steps = 1, (42, 43), (42,), 32, 4
        if a.out == os.path.join(ROOT, d.out_dir):
            cfg.out_dir = os.path.join(ROOT, "outputs_smoke")

    if a.baselines_only:
        from metro_traffic.baselines import baseline_predictions
        from metro_traffic.data import load_processed
        from metro_traffic.metrics import all_metrics, mase_scale
        frame, _, _, _ = load_processed(cfg)
        tr = frame[frame["split"] == "train"]
        scale = mase_scale(tr["traffic_volume"], tr["lag_24"])
        y = frame.loc[frame["split"] == "test", "traffic_volume"].to_numpy(float)
        import pandas as pd
        rows = {n: all_metrics(y, p_, scale) for n, p_ in baseline_predictions(frame, "test").items()}
        print(pd.DataFrame(rows).T.round(3).to_string())
        return

    from metro_traffic.pipeline import Experiment
    table = Experiment(cfg).run_all()
    print("\nTest results (DL rows = mean of the seed models)\n", table.round(3).to_string())


if __name__ == "__main__":
    main()
