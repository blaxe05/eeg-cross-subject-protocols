"""Pinned LibEER model constructors; existing DE-MLP is a local reference."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def make_model(name: str, root: Path) -> torch.nn.Module:
    if name == "DE_MLP":
        from src.strict_eeg_benchmark.phase3_models import MLPDE
        return MLPDE(0.25)
    library = root / "tmp/p3_references/LibEER/LibEER"
    if not library.exists():
        raise FileNotFoundError(library)
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    if name == "DGCNN":
        from models.DGCNN import DGCNN
        return DGCNN(62, 5, 3)
    if name == "HSLT":
        from models.HSLT import HSLT
        return HSLT(62, 5, 3)
    if name == "CDCN":
        from models.CDCN import CDCN
        # The released SEED runner reads config/model_param/CDCN.yaml from its
        # working directory (dropout=.5). Pass the same value explicitly so
        # invocation from this isolated package does not fall back to .4.
        return CDCN(62, 5, 3, dropout=0.5)
    raise ValueError(f"Unrecognized P3 model {name}")


def logits(model: torch.nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "DE_MLP":
        return model(x.flatten(1))[0]
    # HSLT's released forward returns a softmax tensor; its released runner
    # feeds it to CrossEntropyLoss. Preserve that behavior for the bridge.
    return model(x)
