"""Guard the participant unit when training seeds are repeated."""
import importlib.util
from pathlib import Path

import numpy as np


def test_seed_repeats_are_averaged_before_subject_bootstrap():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_controlled_revision.py"
    spec = importlib.util.spec_from_file_location("controlled_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Within-person seed variation is large; only two person means (1 and 3)
    # belong in the descriptive participant-bootstrap distribution.
    result = module.summarize({"person_a": {1: -100., 2: 102.},
                               "person_b": {1: -100., 2: 106.}}, [1, 2],
                              {"bootstrap_seed": 20261029, "bootstrap_replicates": 20000})
    assert result["n_target_people"] == 2
    assert result["mean_pp"] == 2
    assert np.allclose(result["descriptive_ci95_pp"], [1, 3])
    assert result["target_seed_mean_values_pp"] == {"person_a": 1., "person_b": 3.}


def test_public_folds_work_without_private_results(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_controlled_revision.py"
    spec = importlib.util.spec_from_file_location("controlled_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset, target, n_train, n_validation in [
        ("SEED", "1", 11, 3), ("SEED-IV", "1", 11, 3), ("FACED", "sub000", 110, 12)
    ]:
        fold = module.archived_fold(tmp_path / "no_private_results", dataset, target)
        assert len(fold["train"]) == n_train
        assert len(fold["validation"]) == n_validation
        assert target not in set(fold["train"]) | set(fold["validation"])
        assert not set(fold["train"]) & set(fold["validation"])
