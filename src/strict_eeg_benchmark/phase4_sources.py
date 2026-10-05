"""Reconstruct frozen Phase-3 branch outputs for authorized source partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import write_json
from .phase3 import _indices, config_digest
from .phase3_models import build_model
from .phase4_core import assert_partition
from .types import FeatureBatch, FoldSubjects


def _sample_train(indices: np.ndarray, subjects: np.ndarray, per_subject: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for subject in sorted(set(map(str, subjects[indices])), key=int):
        own = indices[np.asarray(subjects[indices], dtype=str) == subject]
        chunks.append(np.sort(rng.choice(own, min(len(own), per_subject), replace=False)))
    return np.sort(np.concatenate(chunks))


@torch.inference_mode()
def _infer_partition(model, model_name: str, batch: FeatureBatch, raw: np.memmap | None, indices: np.ndarray,
                     prep: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, embeddings = [], []
    batch_size = 256 if model_name == "temporal_cnn" else 1024
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        if model_name == "temporal_cnn":
            if raw is None:
                raise ValueError("Temporal CNN requires the Phase-3 raw cache")
            block = np.asarray(raw[selected], dtype=np.float32)
            mean = np.asarray(prep["channel_mean"], dtype=np.float32)[None, :, None]
            std = np.asarray(prep["channel_std"], dtype=np.float32)[None, :, None]
            matrix = (block - mean) / std
        elif model_name == "mlp_de":
            mean = np.asarray(prep["mean"], dtype=np.float32)
            scale = np.asarray(prep["scale"], dtype=np.float32)
            matrix = (np.asarray(batch.X[selected], dtype=np.float32) - mean) / scale
        else:
            raise ValueError(f"Only Phase-4 primary branches can be exported: {model_name}")
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite frozen-branch input")
        result, embedding = model(torch.from_numpy(np.ascontiguousarray(matrix)).to(device))
        logits.append(result.cpu().numpy().astype(np.float32))
        embeddings.append(embedding.cpu().numpy().astype(np.float32))
    z = np.concatenate(embeddings)
    l = np.concatenate(logits)
    if z.shape != (len(indices), 128) or l.shape != (len(indices), 3):
        raise AssertionError("Frozen branch output dimensions changed")
    probabilities = torch.softmax(torch.from_numpy(l), dim=1).numpy()
    return l, probabilities, z


def _target_from_saved(checkpoint: dict, model_name: str, fold_dir: Path, target: str, expected_window_ids: np.ndarray) -> dict:
    with np.load(fold_dir / "target_embeddings.npz", allow_pickle=False) as saved:
        result = {key: np.asarray(saved[key]) for key in ("window_id", "subject_id", "true_label", "class_probabilities", "embedding")}
    if set(map(str, result["subject_id"])) != {target} or not np.array_equal(result["window_id"], expected_window_ids):
        raise AssertionError("Saved target branch has changed window identity or subject")
    if result["embedding"].shape != (len(expected_window_ids), 128):
        raise AssertionError("Target embedding shape differs from Phase 3")
    weights = checkpoint["state_dict"]["head.weight"].numpy()
    bias = checkpoint["state_dict"]["head.bias"].numpy()
    logits = result["embedding"] @ weights.T + bias
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    if not np.allclose(probabilities, result["class_probabilities"], atol=2e-5, rtol=2e-5):
        raise AssertionError(f"{model_name} saved target probabilities do not match frozen head and embeddings")
    result["logits"] = logits.astype(np.float32)
    result["probabilities"] = result.pop("class_probabilities")
    return result


def export_fold_branch(model_name: str, fold: FoldSubjects, batch: FeatureBatch, window_ids: np.ndarray,
                       phase3_root: Path, cache_root: Path, phase3_config: dict, raw: np.memmap | None,
                       per_subject: int, seed: int, device: torch.device) -> dict:
    if model_name not in {"temporal_cnn", "mlp_de"}:
        raise ValueError(model_name)
    train, validation, target = _indices(batch, fold)
    train = _sample_train(train, batch.subject_ids, per_subject, seed)
    assert_partition(batch.subject_ids[train], set(fold.source_train_subjects), fold.held_out_subject, "fusion training export")
    assert_partition(batch.subject_ids[validation], set(fold.source_validation_subjects), fold.held_out_subject, "fusion validation export")
    fold_dir = phase3_root / model_name / "folds" / f"subject_{fold.held_out_subject}"
    diag = json.loads((fold_dir / "diagnostics.json").read_text(encoding="utf-8"))
    if diag["fold"] != fold.to_dict() or diag["config_digest"] != config_digest(phase3_config):
        raise AssertionError("Frozen branch is not the canonical Phase-3 fold")
    if set(diag["preprocessing"]["fit_subjects"]) != set(fold.source_train_subjects):
        raise AssertionError("Branch preprocessing used a non-training subject")
    checkpoint_path = fold_dir / "selected_checkpoint.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["fold"] != fold.to_dict() or checkpoint["model"] != model_name or checkpoint["config_digest"] != diag["config_digest"]:
        raise AssertionError("Checkpoint provenance mismatch")
    out_dir = cache_root / f"subject_{fold.held_out_subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.npz"
    manifest_path = out_dir / f"{model_name}.json"
    expected = {"fold": fold.to_dict(), "model": model_name, "checkpoint_sha256": checkpoint_hash,
                "phase3_config_digest": diag["config_digest"], "source_train_windows_per_subject": per_subject, "sample_seed": seed}
    if path.exists() and manifest_path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise AssertionError("Frozen source cache provenance changed")
        return {"path": path, "metadata": metadata}
    model = build_model(model_name, phase3_config["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    arrays = {}
    for name, indices in (("train", train), ("validation", validation)):
        logits, probabilities, embedding = _infer_partition(model, model_name, batch, raw, indices, diag["preprocessing"], device)
        arrays.update({f"{name}_window_id": window_ids[indices], f"{name}_subject_id": np.asarray(batch.subject_ids[indices], dtype=str),
                       f"{name}_label": np.asarray(batch.y[indices], dtype=np.int8), f"{name}_logits": logits,
                       f"{name}_probabilities": probabilities, f"{name}_embedding": embedding})
    np.savez_compressed(path, **arrays)
    metadata = {**expected, "source_train_subjects": list(fold.source_train_subjects),
                "source_validation_subjects": list(fold.source_validation_subjects),
                "target_subject": fold.held_out_subject, "train_windows": len(train), "validation_windows": len(validation),
                "preprocessing_fit_subjects": diag["preprocessing"]["fit_subjects"],
                "target_source": str((fold_dir / "target_embeddings.npz").resolve())}
    write_json(manifest_path, metadata)
    return {"path": path, "metadata": metadata}


def load_target_branch(model_name: str, fold: FoldSubjects, batch: FeatureBatch, window_ids: np.ndarray,
                       phase3_root: Path, phase3_config: dict) -> dict:
    """Unseal target artifact only after every source-only selection is complete."""
    _, _, target = _indices(batch, fold)
    fold_dir = phase3_root / model_name / "folds" / f"subject_{fold.held_out_subject}"
    checkpoint = torch.load(fold_dir / "selected_checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["fold"] != fold.to_dict() or checkpoint["model"] != model_name or checkpoint["config_digest"] != config_digest(phase3_config):
        raise AssertionError("Target branch checkpoint provenance mismatch")
    result = _target_from_saved(checkpoint, model_name, fold_dir, fold.held_out_subject, window_ids[target])
    if not np.array_equal(result["true_label"], batch.y[target]):
        raise AssertionError("Phase-3 target labels differ from canonical Phase-2 labels")
    return result


def load_source_cache(path: Path, fold: FoldSubjects) -> dict[str, dict]:
    with np.load(path, allow_pickle=False) as saved:
        data = {partition: {name: np.asarray(saved[f"{partition}_{name}"]) for name in
                            ("window_id", "subject_id", "label", "logits", "probabilities", "embedding")}
                for partition in ("train", "validation")}
    for partition, allowed in (("train", set(fold.source_train_subjects)), ("validation", set(fold.source_validation_subjects))):
        assert_partition(data[partition]["subject_id"], allowed, fold.held_out_subject, f"cached {partition} inference")
        if data[partition]["embedding"].shape != (len(data[partition]["label"]), 128):
            raise AssertionError("Cached embedding shape invalid")
    return data
