"""Run isolated LibEER SEED protocol bridge with saved epoch trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p3_protocol_benchmark.data import load_seed_lds
from src.p3_protocol_benchmark.protocol import libeer_9_3_3, loso_14_1, strict_11_3_1
from src.p3_protocol_benchmark.training import run_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("DGCNN", "HSLT", "CDCN", "DE_MLP"), default="DGCNN")
    parser.add_argument("--setting", choices=("libeer_9_3_3", "loso_14_1", "strict_11_3_1"),
                        default="strict_11_3_1")
    parser.add_argument("--normalization", choices=("none", "source_zscore", "per_instance_zscore"),
                        default="none")
    parser.add_argument("--target", type=int, choices=range(1, 16))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.setting == "libeer_9_3_3" and args.target is not None:
        parser.error("Documented 9/3/3 setting has one predetermined three-target split")
    config = json.loads((ROOT / "configs/p3_seed.json").read_text(encoding="utf-8"))
    data = load_seed_lds(ROOT)
    if args.setting == "libeer_9_3_3":
        folds = [libeer_9_3_3(config["seed"])]
    else:
        targets = [str(args.target)] if args.target else list(map(str, range(1, 16)))
        make = (lambda target: loso_14_1(target)) if args.setting == "loso_14_1" else (
            lambda target: strict_11_3_1(target, ROOT))
        folds = [make(target) for target in targets]
    for fold in folds:
        result = run_fold(ROOT, data, config, fold, args.model, args.normalization, smoke=args.smoke)
        print(json.dumps({"model": args.model, "setting": fold.setting, "target": fold.target,
                          "source_epoch": None if result["results"]["source_validation"] is None else
                                          result["results"]["source_validation"]["epoch"],
                          "final_bacc": result["results"]["fixed_final"]["mean_subject_bacc"],
                          "oracle_bacc": result["results"]["target_oracle_diagnostic"]["mean_subject_bacc"],
                          "seconds": result["total_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
