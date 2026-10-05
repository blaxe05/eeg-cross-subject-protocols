"""Run the explicitly target-unlabelled LibEER DANN-DGCNN comparator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p3_protocol_benchmark.data import load_seed_lds
from src.p3_protocol_benchmark.protocol import strict_11_3_1
from src.p3_protocol_benchmark.ta_u import run_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, choices=range(1, 16))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/p3_ta_u.json").read_text(encoding="utf-8"))
    data = load_seed_lds(ROOT)
    for target in ([args.target] if args.target else range(1, 16)):
        fold = strict_11_3_1(str(target), ROOT)
        record = run_fold(ROOT, data, config, fold, smoke=args.smoke)
        print(json.dumps({"target": target, "protocol": "TA-U",
                          "source_epoch": record["results"]["source_validation"]["epoch"],
                          "source_selected_bacc": record["results"]["source_validation"]["mean_subject_bacc"],
                          "oracle_bacc": record["results"]["target_oracle_diagnostic"]["mean_subject_bacc"],
                          "seconds": record["total_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
