"""Explicit subject-level LibEER bridge and strict source-only fold policies."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


SUBJECTS = tuple(map(str, range(1, 16)))


@dataclass(frozen=True)
class Fold:
    setting: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    target: tuple[str, ...]
    sessions: tuple[str, ...] | None

    def validate(self):
        a, b, c = map(set, (self.train, self.validation, self.target))
        if (not a or not c or a & b or a & c or b & c or
                (a | b | c) != set(SUBJECTS)):
            raise AssertionError("P3 subject sets overlap or omit a subject")
        if self.setting in ("strict_11_3_1", "strict_11_3_1_session1") and (len(a), len(b), len(c)) != (11, 3, 1):
            raise AssertionError("P3 strict fold is not 11/3/1")
        if self.setting == "libeer_9_3_3" and (len(a), len(b), len(c)) != (9, 3, 3):
            raise AssertionError("P3 LibEER bridge fold is not 9/3/3")
        if self.setting == "loso_14_1" and (len(a), len(b), len(c)) != (14, 0, 1):
            raise AssertionError("P3 N−1 fold is not 14/0/1")


def libeer_9_3_3(seed=2024):
    """Mirror LibEER's Python-random 20% test, 20% val subject split."""
    shuffled = list(SUBJECTS)
    random.Random(seed).shuffle(shuffled)
    fold = Fold("libeer_9_3_3", tuple(shuffled[6:]), tuple(shuffled[3:6]),
                tuple(shuffled[:3]), None)
    fold.validate()
    return fold


def loso_14_1(target: str, sessions=None):
    fold = Fold("loso_14_1", tuple(s for s in SUBJECTS if s != target), (),
                (target,), None if sessions is None else tuple(map(str, sessions)))
    fold.validate()
    return fold


def strict_11_3_1(target: str, root: Path, sessions=None):
    path = root / "folds/FOLD_MANIFEST.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        matching = [row for row in csv.DictReader(stream)
                    if row["dataset"] == "SEED" and row["task"] == "emotion"
                    and row["target_subject"] == str(target)]
    if len(matching) != 1:
        raise AssertionError(f"Expected one frozen SEED fold for target {target}")
    saved = matching[0]
    fold = Fold("strict_11_3_1", tuple(json.loads(saved["source_train_subjects"])),
                tuple(json.loads(saved["source_validation_subjects"])), (target,),
                None if sessions is None else tuple(map(str, sessions)))
    fold.validate()
    return fold
