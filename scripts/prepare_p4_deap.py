"""Build subject-isolated P4 DEAP DE-LDS features from audited .dat files."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.p4_replication.deap_features import prepare_subject


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=range(1, 33))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for subject in ([args.subject] if args.subject else range(1, 33)):
        print(prepare_subject(root, subject), flush=True)
