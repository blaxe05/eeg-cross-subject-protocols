"""Frozen source-only LOSO assignments for P4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(folds: list[dict], expected_subjects: set[str], n_train: int, n_val: int) -> None:
    if len(folds) != len(expected_subjects):
        raise AssertionError("Incorrect LOSO fold count")
    targets = []
    for fold in folds:
        target = str(fold["held_out_subject"])
        train = set(map(str, fold["source_train_subjects"]))
        validation = set(map(str, fold["source_validation_subjects"]))
        if (len(train) != n_train or len(validation) != n_val or
                train & validation or target in train | validation or
                train | validation | {target} != expected_subjects):
            raise AssertionError(f"Leaking or incomplete fold for {target}")
        targets.append(target)
    if set(targets) != expected_subjects or len(targets) != len(set(targets)):
        raise AssertionError("Target subject coverage is incomplete")


def _deap_folds(audit: dict, seed: int = 20261003) -> list[dict]:
    """Balance source-only validation prevalence for both predeclared tasks."""
    subjects = [f"{i:02d}" for i in range(1, 33)]
    prevalence = {
        task: {sid: int(counts["1"]) / 40
               for sid, counts in audit["tasks"][task]["per_subject"].items()}
        for task in ("valence", "arousal")
    }
    folds = []
    for target in subjects:
        source = [sid for sid in subjects if sid != target]
        source_rates = np.asarray([[prevalence[task][sid] for task in ("valence", "arousal")]
                                   for sid in source])
        source_mean = source_rates.mean(0)
        rng = np.random.default_rng(seed + int(target))
        best = None
        for _ in range(5000):
            candidate = tuple(sorted(rng.choice(len(source), size=6, replace=False).tolist()))
            validation_mean = source_rates[list(candidate)].mean(0)
            train_mean = (source_rates.sum(0) - source_rates[list(candidate)].sum(0)) / 25
            score = float(np.max(np.abs(validation_mean - source_mean)) +
                          np.max(np.abs(train_mean - source_mean)))
            choice = (score, candidate)
            if best is None or choice < best:
                best = choice
        chosen = {source[i] for i in best[1]}
        folds.append({
            "held_out_subject": target,
            "source_train_subjects": [sid for sid in source if sid not in chosen],
            "source_validation_subjects": sorted(chosen),
            "source_prevalence": {task: float(source_mean[k]) for k, task in enumerate(("valence", "arousal"))},
            "source_validation_prevalence": {
                task: float(np.mean([prevalence[task][sid] for sid in chosen]))
                for task in ("valence", "arousal")},
            "source_train_prevalence": {
                task: float(np.mean([prevalence[task][sid] for sid in source if sid not in chosen]))
                for task in ("valence", "arousal")},
            "selection_seed": seed + int(target),
            "selection_rule": "minimum maximum source-only valence/arousal prevalence deviation over 5000 fixed RNG candidates",
        })
    _validate(folds, set(subjects), 25, 6)
    return folds


def freeze_manifests(root: Path) -> Path:
    base = root / "experiments/p4_cross_dataset"
    base.mkdir(parents=True, exist_ok=True)
    output = base / "P4_FOLD_MANIFESTS.json"
    if output.exists():
        raise FileExistsError(f"P4 manifest is frozen: {output}")
    audit_path = base / "DEAP_DATA_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["subjects"] != 32:
        raise AssertionError("Complete DEAP audit is required")
    references = {
        "SEED-IV": root / "experiments/r1_seed_iv/predeclared_folds.json",
        "FACED": root / "experiments/r2_faced/predeclared_folds.json",
    }
    inherited = {}
    for dataset, path in references.items():
        ref = json.loads(path.read_text(encoding="utf-8"))
        folds = ref["folds"]
        subjects = ({str(i) for i in range(1, 16)} if dataset == "SEED-IV" else
                    {f"sub{i:03d}" for i in range(123)})
        _validate(folds, subjects, 11 if dataset == "SEED-IV" else 110,
                  3 if dataset == "SEED-IV" else 12)
        inherited[dataset] = {"source": str(path.relative_to(root)), "sha256": _sha256(path),
                              "folds": folds}
    result = {
        "status": "frozen_before_P4_training",
        "SEED": {"source": "experiments/p3_seed; validated frozen P3 records"},
        **inherited,
        "DEAP": {"source_audit": str(audit_path.relative_to(root)),
                 "source_audit_sha256": _sha256(audit_path),
                 "valence_arousal_shared_folds": _deap_folds(audit)},
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
