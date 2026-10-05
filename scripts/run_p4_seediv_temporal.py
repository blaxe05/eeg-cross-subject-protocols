"""Run one frozen-family SEED-IV R1 TemporalCNN P4 trajectory."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.p4_replication.seediv_temporal import run_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_fold(root, args.target, smoke=args.smoke)
    print({"target": args.target, "source_selected_bacc": result["results"]
           ["source_validation"]["window"]["balanced_accuracy"]})
