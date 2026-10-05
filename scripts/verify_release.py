"""Check fixed subject partitions without accessing licensed recordings."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("SEED", "emotion"): (15, 11, 3),
    ("SEED-IV", "emotion"): (15, 11, 3),
    ("FACED", "emotion"): (123, 110, 12),
    ("DEAP", "valence"): (32, 25, 6),
    ("DEAP", "arousal"): (32, 25, 6),
}


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify() -> None:
    folds = rows("folds/FOLD_MANIFEST.csv")
    assert len(folds) == sum(item[0] for item in EXPECTED.values())
    seen: set[tuple[str, str, str]] = set()
    per_task: dict[tuple[str, str], set[str]] = {key: set() for key in EXPECTED}
    for fold in folds:
        key = (fold["dataset"], fold["task"])
        assert key in EXPECTED, key
        count, n_train, n_val = EXPECTED[key]
        target = fold["target_subject"]
        train = json.loads(fold["source_train_subjects"])
        val = json.loads(fold["source_validation_subjects"])
        assert len(train) == n_train and len(val) == n_val
        assert len(set(train)) == n_train and len(set(val)) == n_val
        assert not (set(train) & set(val))
        assert target not in train and target not in val
        assert len(set(train) | set(val) | {target}) == count
        assert fold["outer_target"] == target
        identity = (*key, target)
        assert identity not in seen
        seen.add(identity)
        per_task[key].add(target)
    assert all(len(per_task[key]) == spec[0] for key, spec in EXPECTED.items())

    print(f"Verified {len(folds)} disjoint outer folds across {len(EXPECTED)} dataset/tasks.")


if __name__ == "__main__":
    verify()
