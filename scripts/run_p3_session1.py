"""Run the same strict 11/3/1 fold on session 1 only for the split bridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p3_protocol_benchmark.data import load_seed_lds
from src.p3_protocol_benchmark.protocol import Fold, strict_11_3_1
from src.p3_protocol_benchmark.training import run_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("DGCNN", "HSLT", "CDCN"), required=True)
    parser.add_argument("--target", type=int, choices=range(1, 16))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/p3_seed_session1.json").read_text(encoding="utf-8"))
    data = load_seed_lds(ROOT)
    for target in ([args.target] if args.target else range(1, 16)):
        reference = strict_11_3_1(str(target), ROOT)
        fold = Fold("strict_11_3_1_session1", reference.train, reference.validation,
                    reference.target, ("1",))
        result = run_fold(ROOT, data, config, fold, args.model, "none", smoke=args.smoke)
        print(json.dumps({"model": args.model, "target": target, "sessions": [1],
                          "source_bacc": result["results"]["source_validation"]["mean_subject_bacc"],
                          "fixed_bacc": result["results"]["fixed_final"]["mean_subject_bacc"],
                          "seconds": result["total_wall_seconds"]}), flush=True)
