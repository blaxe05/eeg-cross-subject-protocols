"""Run one P4 SEED-IV/FACED external-model fold."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.p4_replication.provider_training import load_panel, run_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("SEED-IV", "FACED"), required=True)
    parser.add_argument("--model", choices=("DGCNN", "CDCN"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--setting", choices=("strict", "all_source"), default="strict")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    panel = load_panel(root, args.dataset)
    result = run_fold(root, panel, args.dataset, args.model, args.target, args.setting,
                      smoke=args.smoke)
    print({"dataset": args.dataset, "model": args.model, "target": args.target,
           "setting": args.setting, "fixed_final_bacc": result["results"]["fixed_final"]
           ["window"]["balanced_accuracy"]})
