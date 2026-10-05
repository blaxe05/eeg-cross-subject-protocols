from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strict_eeg_benchmark.artifacts import write_json
from src.strict_eeg_benchmark.audit import audit_seed
from src.strict_eeg_benchmark.datasets import SEEDDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a benchmark dataset without fitting any model")
    parser.add_argument("--dataset", choices=["seed"], required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "SEED")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "audits" / "seed_manifest.json")
    args = parser.parse_args()
    manifest = audit_seed(SEEDDataset(args.data_root))
    write_json(args.output, manifest)
    print(f"Wrote dataset audit manifest: {args.output}")
    print(
        f"subjects={manifest['number_of_subjects']} sessions={manifest['number_of_subject_session_recordings']} "
        f"trials={manifest['number_of_trials']} windows={manifest['total_windows']} "
        f"invalid_values={manifest['invalid_values']}"
    )


if __name__ == "__main__":
    main()
