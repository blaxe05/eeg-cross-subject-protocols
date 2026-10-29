"""Verify local controlled-run partitions, recorded budgets, and scoring arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Production or pilot subdirectory")
    args = parser.parse_args()
    records = sorted(args.run_dir.glob("*/seed_*/target_*/*/result.json"))
    if not records:
        raise AssertionError("No completed cases found")
    verified = 0
    for path in records:
        row = json.loads(path.read_text())
        if row["status"] != "complete":
            raise AssertionError(f"Incomplete record: {path}")
        fold = row["fold"]
        train, validation = set(fold["train"]), set(fold["validation"])
        if train & validation or fold["target"] in train | validation:
            raise AssertionError("Subject crossed fit/validation/target boundaries")
        stage = path.parent.name
        if stage == "checkpoint":
            with np.load(path.parent / "scoring_arrays.npz", allow_pickle=False) as z:
                scores = z["validation_bacc"]
                probability = z["target_probabilities"]
                truth = z["target_labels"]
            if scores.shape != (len(validation), len(row["candidate_steps"])):
                raise AssertionError("Validation person/candidate dimensions disagree")
            curve = np.array([balanced_accuracy_score(truth, p.argmax(1)) for p in probability])
            if not np.allclose(curve, row["target_bacc_curve"], atol=1e-12):
                raise AssertionError("Target probabilities do not reproduce recorded BAcc")
            chosen = scores.argmax(1)
            gains = 100 * (curve[chosen] - curve[-1])
            own = 100 * (scores[np.arange(len(chosen)), chosen] - scores[:, -1])
            if not np.allclose(gains, [r["target_gain_vs_final_pp"] for r in row["rotations"]]):
                raise AssertionError("Checkpoint transfer does not use the common final baseline")
            if not np.allclose(own, [r["selector_gain_vs_final_pp"] for r in row["rotations"]]):
                raise AssertionError("Selection-person gain disagrees")
            if not np.isclose(gains.mean(), row["mean_target_gain_vs_final_pp"]):
                raise AssertionError("Selector rotations were aggregated incorrectly")
            if row["examples_processed"] != row["updates"] * 256:
                raise AssertionError("Checkpoint batch exposure does not match planned budget")
        elif stage == "source":
            base_path = path.parent.parent / "checkpoint/result.json"
            if not base_path.exists():
                raise AssertionError("Source control has no paired strict baseline")
            base = json.loads(base_path.read_text())
            a, b = row["arms"]
            if (a["initial_encoder_sha256"] != base["initial_encoder_sha256"]
                    or b["initial_encoder_sha256"] != base["initial_encoder_sha256"]):
                raise AssertionError("Source controls changed model initialization")
            if a["training_examples"] != base["training_examples"] or a["class_counts"] != base["class_counts"]:
                raise AssertionError("Source composition contrast changes class/example counts")
            if a["updates"] != [base["updates"]] or b["updates"][0] != base["updates"]:
                raise AssertionError("Source controls do not match updates")
            first = np.load(path.parent / "broader_matched/source_rows.npy")
            second = np.load(path.parent / "broader_expanded/source_rows.npy")
            if not set(first) <= set(second) or len(first) != a["training_examples"] or len(second) != b["training_examples"]:
                raise AssertionError("Expanded source pool is not a nested superset")
            proportions = np.array(base["class_counts"]) / base["training_examples"]
            if np.any(abs(np.array(b["class_counts"]) - proportions * b["training_examples"]) > 1 + 1e-8):
                raise AssertionError("Expanded pool changes class proportions beyond rounding")
            if a["training_people"] != len(train | validation) or b["training_people"] != len(train | validation):
                raise AssertionError("Broader source population size changed across controls")
        elif stage == "adaptation":
            arms = row["arms"]
            reference = arms[0]
            for arm in arms:
                for key in ["source_examples", "auxiliary_examples", "source_updates", "source_people",
                            "examples_per_batch", "initial_encoder_sha256", "gradient_reversal_alpha"]:
                    if arm[key] != reference[key]:
                        raise AssertionError(f"Adaptation control changes {key}")
                if arm["target_labels_used_in_fitting"]:
                    raise AssertionError("Target emotion labels entered fitting")
            if fold["target"] in arms[1]["auxiliary_people"] or arms[2]["auxiliary_people"] != [fold["target"]]:
                raise AssertionError("Proxy and target auxiliary populations are not separated")
            if arms[1]["domain_loss_weight"] != arms[2]["domain_loss_weight"]:
                raise AssertionError("Target-access contrast changes domain-loss weight")
            if len(arms) == 4 and not row["loss_off_target_control_passed"]:
                raise AssertionError("Target-free loss-off negative control failed")
        else:
            raise AssertionError(f"Unknown stage: {stage}")
        verified += 1
    print(f"PASS: {verified} completed cases; partitions, budgets, and available scoring arrays verified")


if __name__ == "__main__":
    main()
