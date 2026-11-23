"""Export result figures (PNG), tables (CSV) and a markdown summary into result/.

    python export_results.py                      # reads outputs/, writes result/
    python export_results.py --outputs outputs --result result
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metro_traffic.artifacts import load_artifacts  # noqa: E402
from metro_traffic.report import export_results  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--result", default=os.path.join(ROOT, "result"))
    args = p.parse_args()
    A = load_artifacts(args.outputs)
    if A["results"] is None or A["predictions"] is None:
        sys.exit(f"No results found in {args.outputs}. Run train.py or the notebook first.")
    files = export_results(A, args.result)
    print(f"{len(files)} files written to {os.path.abspath(args.result)}")
    for f in files:
        print("  ", os.path.relpath(f, args.result))
