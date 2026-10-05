from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strict_eeg_benchmark.datasets import SEEDDataset
from src.strict_eeg_benchmark.splits import assert_strict_loso_fold, make_loso_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strict source-only LOSO-DG folds")
    parser.add_argument("--dataset", choices=["seed"], required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "SEED")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    dataset = SEEDDataset(args.data_root)
    all_subjects = set(dataset.subject_ids)
    folds = make_loso_folds(dataset.subject_ids, args.seed, args.validation_fraction)
    seen_targets: set[str] = set()
    for fold in folds:
        assert_strict_loso_fold(fold, all_subjects)
        seen_targets.add(fold.held_out_subject)
        print(
            f"target={fold.held_out_subject:>2} train={','.join(fold.source_train_subjects)} "
            f"validation={','.join(fold.source_validation_subjects)}"
        )
    assert seen_targets == all_subjects, "Each subject must be held out exactly once"
    print(f"PASS: {len(folds)} strict LOSO folds; target subjects never enter train or validation")


if __name__ == "__main__":
    main()
