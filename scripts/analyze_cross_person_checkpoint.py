"""Measure transfer of checkpoint selection between reserved and target people.

Requires locally generated trajectories and licensed dataset caches. The script
does not train models or place data, checkpoints, or results in this repository.
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


def run_case(project: Path, data_parent: Path, output: Path,
             dataset: str, model_name: str, limit_folds: int) -> dict:
    sys.path.insert(0, str(project))
    from src.p3_protocol_benchmark.data import load_seed_lds
    from src.p3_protocol_benchmark.training import _predict as predict_seed
    from src.tac_revision.provider_training import _predict as predict_seediv, load_panel
    from src.tac_revision.models import make_model, CorrectedCDCN

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset == "SEED":
        panel = load_seed_lds(data_parent)
        runs = project / "experiments/p3_seed/runs/strict_11_3_1_session1" / model_name / "none"
        model = make_model(model_name, project).to(device)
    else:
        panel = load_panel(project, "SEED-IV")
        runs = project / "experiments/p4_cross_dataset/runs/SEEDIV" / model_name / "strict"
        library = project / "tmp/p3_references/LibEER/LibEER"
        sys.path.insert(0, str(library))
        if model_name == "DGCNN":
            from models.DGCNN import DGCNN
            model = DGCNN(62, 5, 4).to(device)
        else:
            model = CorrectedCDCN(62, 5, 4, dropout=.5).to(device)

    rotations = []
    provenance = []
    for target in range(1, limit_folds + 1):
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
        if (record["model"] != model_name or actual_target != [str(target)]
                or len(validation) != 3 or set(train) & set(validation)
                or str(target) in set(train) | set(validation)):
            raise AssertionError(f"Fold partition mismatch: {diagnostic}")
        curve = np.asarray(record["target_scores_posthoc"], dtype=np.float64)
        if len(curve) != 40 or not np.isfinite(curve).all():
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
        scores = np.empty((3, 40), dtype=np.float64)
        for epoch in range(1, 41):
            state = torch.load(folder / "trajectory" / f"epoch_{epoch:03d}.pt",
                               map_location=device, weights_only=True)
            model.load_state_dict(state)
            for i, person in enumerate(validation):
                rows = rows_by_person[person]
                if dataset == "SEED":
                    probabilities = predict_seed(model, model_name, panel.x[rows], device)
                    scores[i, epoch - 1] = balanced_accuracy_score(
                        panel.y[rows], probabilities.argmax(1))
                else:
                    probabilities = predict_seediv(model, panel.x[rows], device)
                    scores[i, epoch - 1] = recall_score(
                        panel.y[rows], probabilities.argmax(1),
                        labels=[0, 1, 2, 3], average="macro", zero_division=0)
        for i, person in enumerate(validation):
            selected = earliest_maximum(scores[i])
            comparator = earliest_maximum(scores[np.arange(3) != i].mean(axis=0))
            rotations.append({
                "target_person": str(target), "selector_person": person,
                "comparator_people": ";".join(p for p in validation if p != person),
                "selector_epoch": selected, "comparator_epoch": comparator,
                "selector_person_gain_pp": 100 * float(scores[i, selected - 1] - scores[i, comparator - 1]),
                "independent_target_gain_pp": 100 * float(curve[selected - 1] - curve[comparator - 1]),
            })
        provenance.append({"target_person": str(target), "diagnostics": str(diagnostic),
                           "train_people": train, "validation_people": validation})
        print(f"{dataset} {model_name} target {target}/{limit_folds}", flush=True)

    subject_rows = []
    for target in range(1, limit_folds + 1):
        subset = [r for r in rotations if r["target_person"] == str(target)]
        if len(subset) != 3:
            raise AssertionError("Missing selector rotation")
        subject_rows.append({
            "target_person": str(target),
            "mean_selector_person_gain_pp": float(np.mean([r["selector_person_gain_pp"] for r in subset])),
            "mean_independent_target_gain_pp": float(np.mean([r["independent_target_gain_pp"] for r in subset])),
        })
    apparent = np.array([r["mean_selector_person_gain_pp"] for r in subject_rows])
    independent = np.array([r["mean_independent_target_gain_pp"] for r in subject_rows])
    summary = {"n_outer_target_people": limit_folds,
               "n_selector_rotations": len(rotations),
               "mean_selector_person_gain_pp": float(apparent.mean()),
               "mean_independent_target_gain_pp": float(independent.mean()),
               "by_target_person": subject_rows}
    if limit_folds == 15:
        summary["descriptive_subject_bootstrap_ci95_pp"] = bootstrap_interval(independent)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.replace('-', '').lower()}_{model_name.lower()}_cross_person"
    result = {"dataset": dataset, "model": model_name, "status": "complete" if limit_folds == 15 else "smoke_only",
              "protocol": "one reserved validation person selects; other two select comparator; separate outer target scores",
              "archived_runs": str(runs), "folds": provenance, "summary": summary}
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rotations[0]))
        writer.writeheader()
        writer.writerows(rotations)
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-parent", type=Path, help="Directory containing data/SEED; defaults to --project-root")
    parser.add_argument("--output-dir", type=Path, help="Local ignored output directory")
    parser.add_argument("--dataset", choices=["SEED", "SEED-IV"], required=True)
    parser.add_argument("--model", choices=["DGCNN", "CDCN"], required=True)
    parser.add_argument("--limit-folds", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.limit_folds <= 15:
        raise ValueError("--limit-folds must be 1..15")
    project = args.project_root.resolve()
    output = args.output_dir or project / "experiments/cross_person_checkpoint"
    run_case(project, (args.data_parent or project).resolve(), output.resolve(),
             args.dataset, args.model, args.limit_folds)


if __name__ == "__main__":
    main()
