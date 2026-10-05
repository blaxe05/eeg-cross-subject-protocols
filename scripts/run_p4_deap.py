"""Run one P4 DEAP DGCNN fold, with an optional tiny engineering smoke."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.p4_replication.deap_training import run_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("valence", "arousal"), required=True)
    parser.add_argument("--target", choices=[f"{i:02d}" for i in range(1, 33)], required=True)
    parser.add_argument("--setting", choices=("strict_25_6_1", "all_source_31_1"),
                        default="strict_25_6_1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    record = run_fold(Path(__file__).resolve().parents[1], args.task, args.target,
                      args.setting, smoke=args.smoke)
    print({"task": record["task"], "target": record["target_subject"],
           "setting": record["setting"], "source_epoch": record["results"]["source_validation"]
           and record["results"]["source_validation"]["epoch"],
           "fixed_final_bacc": record["results"]["fixed_final"]["window"]["balanced_accuracy"]})
