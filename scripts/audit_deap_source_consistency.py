"""Audit DEAP subject ratings against local participant metadata without edits.

The per-trial output can contain access-restricted metadata and must stay local.
Agreement or a recognizable transformation does not authenticate file origin.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workbook = data / "Metadata/participant_ratings.xls"
    second = data / "metadata_xls/participant_ratings.xls"
    frame = pd.read_excel(workbook)
    if len(frame) != 1280 or frame.duplicated(["Participant_id", "Experiment_id"]).any():
        raise AssertionError("Participant metadata grain or completeness is invalid")
    fields = ["Valence", "Arousal", "Dominance", "Liking"]
    rows, subjects = [], []
    hashes = {str(workbook): sha256(workbook), str(second): sha256(second)}
    for sid in range(1, 33):
        reference = frame[frame.Participant_id == sid].sort_values("Experiment_id")
        if reference.Experiment_id.tolist() != list(range(1, 41)):
            raise AssertionError("Subject trial IDs are incomplete")
        expected = reference[fields].to_numpy(dtype=np.float64)
        if not np.isfinite(expected).all() or np.any((expected < 1) | (expected > 9)):
            raise AssertionError("Workbook ratings violate the documented scale")
        path = data / "data" / f"s{sid:02d}.dat"
        hashes[str(path)] = sha256(path)
        with path.open("rb") as handle:
            record = pickle.load(handle, encoding="latin1")
        local = np.asarray(record["labels"], dtype=np.float64)
        if local.shape != (40, 4) or not np.isfinite(local).all():
            raise AssertionError("Subject rating array shape or finiteness is invalid")
        if np.asarray(record["data"]).shape != (40, 40, 8064):
            raise AssertionError("Subject EEG array shape changed")
        mismatch = ~np.isclose(local, expected, rtol=0, atol=1e-10)
        high_high = (expected[:, 0] > 5) & (expected[:, 1] > 5)
        transformed = expected.copy()
        transformed[high_high, :2] = 9 - transformed[high_high, :2]
        unexplained = ~np.isclose(local, transformed, rtol=0, atol=1e-10)
        invalid = (local < 1) | (local > 9)
        subjects.append({"subject": f"s{sid:02d}", "changed_trials": int(mismatch.any(1).sum()),
            "invalid_rating_cells": int(invalid.sum()), "unexplained_cells": int(unexplained.sum()),
            "dominance_liking_mismatches": int(mismatch[:, 2:].sum())})
        for i in range(40):
            row = {"subject": f"s{sid:02d}", "experiment_id": i + 1,
                "presentation_trial": int(reference.iloc[i]["Trial"]),
                "changed": bool(mismatch[i].any()), "workbook_high_high": bool(high_high[i]),
                "exact_nine_minus_pattern": bool(not unexplained[i].any()),
                "invalid_local_cells": int(invalid[i].sum())}
            for j, field in enumerate(fields):
                row["workbook_" + field.lower()] = float(expected[i, j])
                row["dat_" + field.lower()] = float(local[i, j])
            rows.append(row)
        del record
    for filename, records in [("all_trial_comparison.csv", rows),
                              ("affected_trial_ids.csv", [r for r in rows if r["changed"]]),
                              ("subject_summary.csv", subjects)]:
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {"subjects": 32, "trials": len(rows),
        "changed_trials": sum(r["changed"] for r in rows),
        "workbook_high_high_trials": sum(r["workbook_high_high"] for r in rows),
        "invalid_local_rating_cells": sum(r["invalid_local_cells"] for r in rows),
        "unexplained_cells_after_nine_minus_high_high": sum(s["unexplained_cells"] for s in subjects),
        "dominance_liking_mismatches": sum(s["dominance_liking_mismatches"] for s in subjects),
        "workbook_copies_byte_identical": hashes[str(workbook)] == hashes[str(second)],
        "class_counts": {source: {dimension: {
            "low": sum(r[f"{source}_{dimension}"] <= 5 for r in rows),
            "high": sum(r[f"{source}_{dimension}"] > 5 for r in rows)}
            for dimension in ["valence", "arousal"]} for source in ["workbook", "dat"]},
        "file_sha256": hashes, "raw_files_modified": False,
        "authentication": "Local consistency audit only; original provider file identity is not independently authenticated.",
        "interpretation": "Local rating differences exactly match nine-minus-workbook on high-valence/high-arousal trials; origin and cause of the alteration are unknown."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
