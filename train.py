"""Run the full experiment from the command line and write the dashboard artifacts.

Examples
    python train.py                         # full run (GPU recommended)
    python train.py --no-tuning --no-ablation --epochs 20
    python train.py --models LSTM GRU
    python train.py --baselines-only        # no PyTorch needed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from metro_traffic.config import ALL_MODELS, Config  # noqa: E402
from metro_traffic.pipeline import Experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=Config.data_path)
    p.add_argument("--out", default=Config.out_dir)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--trials", type=int, default=Config.n_trials)
    p.add_argument("--no-tuning", action="store_true")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--baselines-only", action="store_true")
    args = p.parse_args()

    cfg = Config(data_path=args.data, out_dir=args.out, models=tuple(args.models), epochs=args.epochs,
                 n_trials=args.trials, run_tuning=not args.no_tuning, run_ablation=not args.no_ablation)
    exp = Experiment(cfg)
    if args.baselines_only:
        exp.load_data()
        exp.run_baselines()
        exp.save()
        print(exp.tracker.table().round(3).to_string())
        return
    table = exp.run_all()
    print("\nTest results\n", table.round(3).to_string())


if __name__ == "__main__":
    main()
