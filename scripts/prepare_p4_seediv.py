"""Build P4's provider-provenance SEED-IV external-model input cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.p4_replication.seediv_features import build_seediv_cache


if __name__ == "__main__":
    print(build_seediv_cache(Path(__file__).resolve().parents[1]))
