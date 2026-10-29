"""Measure transfer of checkpoint selection between reserved and target people.

Requires locally generated trajectories and licensed dataset caches. The script
uses archived weights; generated results stay local and are ignored by Git.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score


def earliest_maximum(values: np.ndarray) -> int:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Checkpoint scores must be one-dimensional and finite")
    return int(np.argmax(values)) + 1


def bootstrap_interval(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20261029)
    indices = rng.integers(0, len(values), size=(20000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def load_faced_panel(project: Path, data_parent: Path):
    """Load the frozen FACED feature cache and audited subject/window index."""
    from src.strict_eeg_benchmark.datasets import FACEDDataset
    from src.strict_eeg_benchmark.r2_data import build_window_index
    from src.tac_revision.provider_training import Panel

    dataset = FACEDDataset(data_parent / "data/FACED",
                           project / "artifacts/audits/faced_manifest.json")
    index = build_window_index(dataset, project / "artifacts/audits/faced_trials.csv")
    cache = project / "experiments/r2d_faced/cache/de_5e82d3fbcad53225.npy"
    metadata = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    x = np.load(cache, mmap_mode="r")
    if (x.shape != (103320, 160)
            or metadata["signature"] != "5e82d3fbcad53225276989937b628f984938783bd8e7f65a71e3b314720c84ff"
            or len(index.y) != len(x)):
        raise AssertionError("Frozen FACED feature cache or index changed")
    return Panel(x.reshape(-1, 32, 5), index.y, index.subject_ids,
                 index.trial_ids, index.local_window, 32, 9, str(cache.relative_to(project)))


def run_case(project: Path, data_parent: Path, output: Path,
             dataset: str, model_name: str, limit_folds: int) -> dict:
    sys.path.insert(0, str(project))
    from src.p3_protocol_benchmark.data import load_seed_lds
    from src.p3_protocol_benchmark.training import _predict as predict_seed
    from src.tac_revision.provider_training import _predict as predict_seediv, load_panel
    from src.tac_revision.models import make_model, CorrectedCDCN

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library = project / "tmp/p3_references/LibEER/LibEER"
    sys.path.insert(0, str(library))
    if dataset == "SEED":
        panel = load_seed_lds(data_parent)
        runs = project / "experiments/p3_seed/runs/strict_11_3_1_session1" / model_name / "none"
        model = make_model(model_name, project).to(device)
        targets = [str(i) for i in range(1, 16)]
        epochs = 40
        expected_validation = 3
    elif dataset == "SEED-IV":
        panel = load_panel(project, "SEED-IV")
        runs = project / "experiments/p4_cross_dataset/runs/SEEDIV" / model_name / "strict"
        if model_name == "DGCNN":
            from models.DGCNN import DGCNN
            model = DGCNN(62, 5, 4).to(device)
        else:
            model = CorrectedCDCN(62, 5, 4, dropout=.5).to(device)
        targets = [str(i) for i in range(1, 16)]
        epochs = 40
        expected_validation = 3
    else:
        if model_name != "DGCNN":
            raise ValueError("The archived FACED extension uses DGCNN only")
        from models.DGCNN import DGCNN
        panel = load_faced_panel(project, data_parent)
        runs = project / "experiments/p4_cross_dataset/runs/FACED/DGCNN/strict"
        model = DGCNN(panel.n_channels, 5, panel.n_classes).to(device)
        targets = sorted(set(panel.subject))
        epochs = 8
        expected_validation = 12
        if len(targets) != 123:
            raise AssertionError("Expected all 123 FACED people")

    rotations = []
    provenance = []
    for target in targets[:limit_folds]:
        folder = runs / f"target_{target}"
        if dataset == "SEED" and model_name == "CDCN":
            diagnostic = (project / "experiments/tac_revision_20261004/cdcn_reevaluation/corrected/SEED"
                          / "strict_11_3_1_session1" / f"target_{target}/diagnostics.json")
        else:
            diagnostic = folder / "diagnostics.json"
        record = json.loads(diagnostic.read_text(encoding="utf-8"))
        train = record.get("train_subjects", record.get("source_train_subjects"))
        validation = record.get("validation_subjects", record.get("source_validation_subjects"))
        actual_target = record.get("target_subjects", [record.get("target_subject")])
        if (record["model"] != model_name or actual_target != [target]
                or len(validation) != expected_validation or set(train) & set(validation)
                or target in set(train) | set(validation)
                or set(train) | set(validation) | {target} != set(panel.subject)):
            raise AssertionError(f"Fold partition mismatch: {diagnostic}")
        if dataset == "FACED" and (record["dataset"] != "FACED" or record["setting"] != "strict"
                or int(record["settings"]["epochs"]) != 8
                or record["target_EEG_used_in_fitting"]
                or record["target_labels_used_in_fitting"]
                or not record["source_selection_completed_before_target_access"]):
            raise AssertionError(f"FACED archive protocol mismatch: {diagnostic}")
        curve = np.asarray(record["target_scores_posthoc"], dtype=np.float64)
        if len(curve) != epochs or not np.isfinite(curve).all():
            raise AssertionError(f"Incomplete target epoch curve: {diagnostic}")
        reference = record["results"]["source_validation"]
        reference_score = (reference["mean_subject_bacc"]
                           if dataset == "SEED" and model_name == "DGCNN"
                           else reference["window"]["balanced_accuracy"])
        if not np.isclose(curve[int(reference["epoch"]) - 1], reference_score, atol=1e-10):
            raise AssertionError(f"Target curve and selected result differ: {diagnostic}")
        if dataset == "SEED":
            rows_by_person = {person: panel.subset((person,), (1,)) for person in validation}
        else:
            rows_by_person = {person: np.flatnonzero(panel.subject == person) for person in validation}
        if any(len(rows) == 0 or set(panel.subject[rows]) != {person}
               for person, rows in rows_by_person.items()):
            raise AssertionError("Validation subject not found in cache")
        scores = np.empty((len(validation), epochs), dtype=np.float64)
        if dataset == "FACED":
            combined_rows = np.concatenate(list(rows_by_person.values()))
            offsets = np.cumsum([0] + [len(rows) for rows in rows_by_person.values()])
        for epoch in range(1, epochs + 1):
            state = torch.load(folder / "trajectory" / f"epoch_{epoch:03d}.pt",
                               map_location=device, weights_only=True)
            model.load_state_dict(state)
            if dataset == "FACED":
                probabilities = predict_seediv(model, panel.x[combined_rows], device)
            for i, person in enumerate(validation):
                rows = rows_by_person[person]
                if dataset == "SEED":
                    probabilities = predict_seed(model, model_name, panel.x[rows], device)
                    scores[i, epoch - 1] = balanced_accuracy_score(
                        panel.y[rows], probabilities.argmax(1))
                else:
                    person_probabilities = (probabilities[offsets[i]:offsets[i + 1]]
                                            if dataset == "FACED"
                                            else predict_seediv(model, panel.x[rows], device))
                    scores[i, epoch - 1] = recall_score(
                        panel.y[rows], person_probabilities.argmax(1),
                        labels=list(range(panel.n_classes)), average="macro", zero_division=0)
        for i, person in enumerate(validation):
            selected = earliest_maximum(scores[i])
            comparator = earliest_maximum(scores[np.arange(len(validation)) != i].mean(axis=0))
            rotations.append({
                "target_person": target, "selector_person": person,
                "comparator_people": ";".join(p for p in validation if p != person),
                "selector_epoch": selected, "comparator_epoch": comparator,
                "selector_person_gain_pp": 100 * float(scores[i, selected - 1] - scores[i, comparator - 1]),
                "independent_target_gain_pp": 100 * float(curve[selected - 1] - curve[comparator - 1]),
                "independent_target_selector_bacc": float(curve[selected - 1]),
                "independent_target_comparator_bacc": float(curve[comparator - 1]),
            })
        provenance.append({"target_person": target, "diagnostics": str(diagnostic),
                           "train_people": train, "validation_people": validation})
        print(f"{dataset} {model_name} target {target}/{limit_folds}", flush=True)

    subject_rows = []
    for target in targets[:limit_folds]:
        subset = [r for r in rotations if r["target_person"] == target]
        if len(subset) != expected_validation:
            raise AssertionError("Missing selector rotation")
        subject_rows.append({
            "target_person": target,
            "mean_selector_person_gain_pp": float(np.mean([r["selector_person_gain_pp"] for r in subset])),
            "mean_independent_target_gain_pp": float(np.mean([r["independent_target_gain_pp"] for r in subset])),
        })
    apparent = np.array([r["mean_selector_person_gain_pp"] for r in subject_rows])
    independent = np.array([r["mean_independent_target_gain_pp"] for r in subject_rows])
    summary = {"n_outer_target_people": limit_folds,
               "n_selector_rotations": len(rotations),
               "n_candidate_epochs": epochs,
               "n_reserved_validation_people": expected_validation,
               "mean_selector_person_gain_pp": float(apparent.mean()),
               "mean_independent_target_gain_pp": float(independent.mean()),
               "independent_target_sign_counts": {
                   "positive": int(np.count_nonzero(independent > 1e-12)),
                   "negative": int(np.count_nonzero(independent < -1e-12)),
                   "equal": int(np.count_nonzero(np.abs(independent) <= 1e-12))},
               "by_target_person": subject_rows}
    if limit_folds == len(targets):
        summary["descriptive_subject_bootstrap_ci95_pp"] = bootstrap_interval(independent)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.replace('-', '').lower()}_{model_name.lower()}_cross_person"
    result = {"dataset": dataset, "model": model_name,
              "status": "complete" if limit_folds == len(targets) else "smoke_only",
              "protocol": "one reserved validation person selects; remaining validation people select comparator; separate outer target scores",
              "archived_runs": str(runs), "folds": provenance, "summary": summary}
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rotations[0]))
        writer.writeheader()
        writer.writerows(rotations)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "by_target_person"}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-parent", type=Path,
                        help="Directory containing data/SEED or data/FACED; defaults to --project-root")
    parser.add_argument("--output-dir", type=Path, help="Local ignored output directory")
    parser.add_argument("--dataset", choices=["SEED", "SEED-IV", "FACED"], required=True)
    parser.add_argument("--model", choices=["DGCNN", "CDCN"], required=True)
    parser.add_argument("--limit-folds", type=int,
                        help="Optional first-fold smoke limit; defaults to all people")
    args = parser.parse_args()
    maximum = 123 if args.dataset == "FACED" else 15
    limit = maximum if args.limit_folds is None else args.limit_folds
    if not 1 <= limit <= maximum:
        raise ValueError(f"--limit-folds must be 1..{maximum}")
    project = args.project_root.resolve()
    output = args.output_dir or project / "experiments/cross_person_checkpoint"
    run_case(project, (args.data_parent or project).resolve(), output.resolve(),
             args.dataset, args.model, limit)


if __name__ == "__main__":
    main()
