"""Guard optional E60 combinations against target-driven family selection."""

from __future__ import annotations

import json
from pathlib import Path

def justified_combination_parameters(output_root: Path, phase3_root: Path, folds, config: dict) -> dict:
    """Stop the unqualified combined experiment under the strict primary protocol."""
    del phase3_root, folds, config
    gate_path = output_root / "conditional_decisions.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate["E60_3"]["execute"]:
        return {}
    raise RuntimeError("E60.3 requires an independent predeclared source-only family decision; "
                       "SEED target-derived research gates cannot authorize a strict primary LOSO estimate")
