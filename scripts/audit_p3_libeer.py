"""Exhaustively verify P3 cache order against all provider SEED de_LDS trials."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p3_protocol_benchmark.data import load_seed_lds


def main():
    data = load_seed_lds(ROOT)
    names, starts = np.unique(data.trial, return_index=True)
    ordered = names[np.argsort(starts)]
    if len(ordered) != 675:
        raise AssertionError("Expected 675 unique SEED trials")
    checked = 0
    maximum_difference = 0.0
    for filename in dict.fromkeys(name.rsplit("_trial_", 1)[0] for name in ordered):
        path = ROOT / "data/SEED/ExtractedFeatures_1s" / f"{filename}.mat"
        matlab = loadmat(path)
        for ordinal in range(1, 16):
            name = f"{filename}_trial_{ordinal:02d}"
            rows = np.flatnonzero(data.trial == name)
            provider = np.transpose(matlab[f"de_LDS{ordinal}"], (1, 0, 2)).astype(np.float32)
            if (provider.shape != data.x[rows].shape or len(rows) == 0 or
                    not np.all(np.diff(rows) == 1)):
                raise AssertionError(f"P3 provider trial order/shape mismatch: {name}")
            difference = float(np.max(np.abs(provider - data.x[rows])))
            maximum_difference = max(maximum_difference, difference)
            if difference > 5e-6:
                raise AssertionError(f"P3 provider feature mismatch: {name} diff={difference}")
            checked += 1
    library = ROOT / "tmp/p3_references/LibEER"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=library, text=True).strip()
    if commit != "dddff9776dbdae21195fe320dff0a5ba61628a18":
        raise AssertionError("LibEER revision changed during P3")
    digest = hashlib.sha256()
    with (ROOT / "data/SEED/seed_lds_cache.npz").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = {"status": "validated", "library_commit": commit,
              "feature_cache_sha256": digest.hexdigest(), "windows": len(data.x),
              "shape": list(data.x.shape), "subjects": 15, "subject_session_pairs": 45,
              "trials": checked, "labels": {str(i): int(np.sum(data.y == i)) for i in range(3)},
              "maximum_provider_abs_difference": maximum_difference,
              "sessions": {str(i): int(np.sum(data.session_ordinal == i)) for i in (1, 2, 3)}}
    output = ROOT / "experiments/p3_seed/P3_DATA_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
