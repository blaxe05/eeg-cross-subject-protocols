"""Run one matched P4 DANN-DGCNN TA-U fold."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.p4_replication.ta_u import run_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("SEED-IV", "FACED", "DEAP-valence", "DEAP-arousal"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    record = run_fold(Path(__file__).resolve().parents[1], args.dataset, args.target,
                      smoke=args.smoke)
    print({"dataset": args.dataset, "target": args.target,
           "source_selected_bacc": record["results"]["source_validation"]
           ["window"]["balanced_accuracy"]})
