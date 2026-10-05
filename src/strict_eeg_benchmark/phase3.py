"""Strict source-only Phase-3 single-view SEED training and artifact validation."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .artifacts import write_json
from .datasets import SEEDDataset
from .evaluation import classification_metrics
from .phase2 import CLASS_NAMES, CLASSES, _session_numbers, aggregate_subject_rows, oof_frame, validate_oof
from .phase3_data import fit_raw_channel_stats, phase2_folds, raw_window_cache, subject_indices
from .phase3_models import ARCHITECTURE_DESCRIPTIONS, MODEL_NAMES, RAW_MODELS, anatomical_adjacency, build_model
from .preprocessing import AuditedStandardScaler
from .types import FeatureBatch, FoldSubjects, Partition


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["phase2_reference_seed"] != 20260928 or set(config["models"]) != set(MODEL_NAMES):
        raise ValueError("Phase-3 configuration differs from the predeclared model/fold specification")
    if config["learning_rates"] != [0.001, 0.0003] or config["checkpoint_metric"] != "source_validation_balanced_accuracy":
        raise ValueError("Phase-3 source-only hyperparameter grid changed")
    return config


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = classification_metrics(y, prediction, CLASSES)
    result["confusion_matrix"] = confusion_matrix(y, prediction, labels=CLASSES).tolist()
    return result


def validate_class_probabilities(frame: pd.DataFrame) -> None:
    probabilities = frame[["score_negative", "score_neutral", "score_positive"]].to_numpy()
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            or not np.array_equal(probabilities.argmax(axis=1), frame["predicted_label"].to_numpy())):
        raise AssertionError("Target probabilities or class decisions are inconsistent")


def _indices(batch: FeatureBatch, fold: FoldSubjects) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = subject_indices(batch, fold.source_train_subjects)
    validation = subject_indices(batch, fold.source_validation_subjects)
    target = subject_indices(batch, {fold.held_out_subject})
    if len(set(train) & set(validation)) or len(set(train) & set(target)) or len(set(validation) & set(target)):
        raise AssertionError("Source/validation/target index overlap")
    if len(train) + len(validation) + len(target) != len(batch.y):
        raise AssertionError("Fold does not cover the Phase-2 window set")
    if set(map(str, batch.subject_ids[train])) != set(fold.source_train_subjects):
        raise AssertionError("Training indices contain unexpected subjects")
    if set(map(str, batch.subject_ids[validation])) != set(fold.source_validation_subjects):
        raise AssertionError("Validation indices contain unexpected subjects")
    if set(map(str, batch.subject_ids[target])) != {fold.held_out_subject}:
        raise AssertionError("Target indices contain unexpected subjects")
    return train, validation, target


def _raw_norm(raw: np.memmap, train: np.ndarray, fold: FoldSubjects, cache_root: Path, signature: str, subject_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if set(map(str, subject_ids[train])) != set(fold.source_train_subjects):
        raise AssertionError("Raw normalization input includes a non-training subject")
    path = cache_root / f"raw_normalization_subject_{fold.held_out_subject}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("raw_signature") != signature or saved.get("fit_subjects") != list(fold.source_train_subjects) or saved.get("fit_window_count") != len(train):
            raise AssertionError("Source-training raw normalization cache provenance mismatch")
        return np.asarray(saved["channel_mean"], dtype=np.float32), np.asarray(saved["channel_std"], dtype=np.float32), saved
    mean, std, metadata = fit_raw_channel_stats(raw, train, fold, subject_ids)
    metadata["raw_signature"] = signature
    write_json(path, metadata)
    return mean, std, metadata


def _tensor_from_raw(raw: np.memmap, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    """Transfer only the requested partition; normalization parameters came from source training."""
    output = torch.empty((len(indices), 62, 200), dtype=torch.float32, device=device)
    mean_tensor = torch.as_tensor(mean, device=device).view(1, 62, 1)
    std_tensor = torch.as_tensor(std, device=device).view(1, 62, 1)
    for start in range(0, len(indices), 1024):
        stop = min(start + 1024, len(indices))
        block = np.array(raw[indices[start:stop]], copy=True)
        output[start:stop] = (torch.from_numpy(block).to(device) - mean_tensor) / std_tensor
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite source-normalized raw input")
    return output


def _tensor_from_de(batch: FeatureBatch, indices: np.ndarray, scaler: AuditedStandardScaler, model_name: str, device: torch.device) -> torch.Tensor:
    matrix = scaler.transform(batch.X[indices]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite source-scaled DE input")
    if model_name != "mlp_de":
        matrix = matrix.reshape(-1, 62, 5)
    return torch.from_numpy(matrix).to(device)


@torch.inference_mode()
def _infer(model: torch.nn.Module, x: torch.Tensor, batch_size: int, embeddings: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    predictions: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        logits, embedding = model(x[start:start + batch_size])
        if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
            raise ValueError("Nonfinite model output")
        predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
        if embeddings:
            vectors.append(embedding.cpu().numpy())
    probabilities = np.concatenate(predictions)
    return probabilities, np.concatenate(vectors) if embeddings else None


def _train_candidate(
    model_name: str, learning_rate: float, config: dict[str, Any], seed: int,
    x_train: torch.Tensor, y_train: torch.Tensor, x_validation: torch.Tensor, y_validation: np.ndarray,
    adjacency: np.ndarray | None, batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    set_determinism(seed)
    model = build_model(model_name, config["dropout"], adjacency).to(x_train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    patience = 0
    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y_train), device=x_train.device)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x_train[rows])
            loss = torch.nn.functional.cross_entropy(logits, y_train[rows])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        probabilities, _ = _infer(model, x_validation, batch_size)
        score = float(balanced_accuracy_score(y_validation, probabilities.argmax(axis=1)))
        history.append({"epoch": epoch, "training_cross_entropy": total_loss / len(y_train), "source_validation_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("No valid source-validation checkpoint")
    return best_state, {
        "learning_rate": learning_rate, "seed": seed, "best_epoch": best_epoch,
        "best_source_validation_balanced_accuracy": best_score,
        "epochs_run": len(history), "history": history,
    }


def _probe_source_subjects(
    model: torch.nn.Module, x_train: torch.Tensor, train_indices: np.ndarray, batch: FeatureBatch, fold: FoldSubjects, batch_size: int,
) -> dict[str, Any]:
    """Frozen-embedding, within-source-training trial split; never influences a checkpoint."""
    subject_values = np.asarray(batch.subject_ids[train_indices], dtype=str)
    if set(subject_values) != set(fold.source_train_subjects) or fold.held_out_subject in subject_values:
        raise AssertionError("Subject probe was given non-training subject embeddings")
    trial_values = np.asarray(batch.trial_ids[train_indices], dtype=str)
    train_local: list[int] = []
    test_local: list[int] = []
    for subject in sorted(set(subject_values), key=int):
        trials = sorted(set(trial_values[subject_values == subject]))
        if len(trials) < 4:
            raise ValueError("Subject probe needs at least four source-training trials")
        held_trials = set(trials[::4])
        source_trials = set(trials) - held_trials
        train_local.extend(np.flatnonzero((subject_values == subject) & np.isin(trial_values, list(source_trials)))[:300])
        test_local.extend(np.flatnonzero((subject_values == subject) & np.isin(trial_values, list(held_trials)))[:100])
    selected = np.asarray(train_local + test_local, dtype=int)
    _, embeddings = _infer(model, x_train[selected], batch_size, embeddings=True)
    assert embeddings is not None
    n_train = len(train_local)
    labels = subject_values[selected]
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    classifier.fit(embeddings[:n_train], labels[:n_train])
    predicted = classifier.predict(embeddings[n_train:])
    return {
        "protocol": "frozen source-training embeddings; every fourth trial held out within each of 11 source-training subjects",
        "training_window_count": n_train, "test_window_count": len(test_local),
        "n_subject_classes": len(set(labels[:n_train])),
        "accuracy": float(accuracy_score(labels[n_train:], predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[n_train:], predicted)),
        "used_for_model_selection": False,
    }


def run_fold(
    dataset: SEEDDataset, batch: FeatureBatch, window_ids: np.ndarray, fold: FoldSubjects,
    fold_index: int, model_name: str, config: dict[str, Any], output_root: Path,
    phase2_root: Path, raw: np.memmap | None = None, graph: tuple[np.ndarray, dict] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = output_root / model_name / "folds" / f"subject_{fold.held_out_subject}"
    digest = config_digest(config)
    diagnostic_path = fold_dir / "diagnostics.json"
    if diagnostic_path.exists():
        saved = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if saved["config_digest"] != digest or saved["fold"] != fold.to_dict():
            raise AssertionError("Existing Phase-3 fold differs from current configuration or Phase-2 fold")
        if not (fold_dir / "target_predictions.csv.gz").exists() or not (fold_dir / "target_embeddings.npz").exists() or not (fold_dir / "selected_checkpoint.pt").exists():
            raise AssertionError("Existing Phase-3 fold is incomplete")
        return saved
    train, validation, target = _indices(batch, fold)
    # Assert the reference fold *before* any model fitting. Phase-2 saved fold is authoritative.
    phase2_saved = json.loads((phase2_root / "raw_de_logistic_regression" / "folds" / f"subject_{fold.held_out_subject}" / "diagnostics.json").read_text(encoding="utf-8"))["fold"]
    if phase2_saved != fold.to_dict():
        raise AssertionError("Phase-3 fold does not match saved Phase-2 reference")
    preprocessing: dict[str, Any]
    if model_name in RAW_MODELS:
        if raw is None:
            raise ValueError("Raw EEG cache required")
        raw_meta = next((output_root / "cache").glob("raw_windows_*.json"))
        signature = json.loads(raw_meta.read_text(encoding="utf-8"))["signature"]
        mean, std, preprocessing = _raw_norm(raw, train, fold, output_root / "cache", signature, batch.subject_ids)
        x_train = _tensor_from_raw(raw, train, mean, std, device)
        x_validation = _tensor_from_raw(raw, validation, mean, std, device)
        batch_size = config["batch_size_raw"]
    else:
        train_batch = batch.subset_subjects(set(fold.source_train_subjects), Partition.SOURCE_TRAIN)
        scaler = AuditedStandardScaler().fit(train_batch, fold)
        preprocessing = scaler.metadata()
        x_train = _tensor_from_de(batch, train, scaler, model_name, device)
        x_validation = _tensor_from_de(batch, validation, scaler, model_name, device)
        batch_size = config["batch_size_de"]
    y_train = torch.as_tensor(batch.y[train], dtype=torch.long, device=device)
    y_validation = np.asarray(batch.y[validation], dtype=np.int64)
    adjacency = graph[0] if graph is not None else None
    candidates: list[dict[str, Any]] = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected: dict[str, Any] | None = None
    for candidate_index, lr in enumerate(config["learning_rates"]):
        candidate_seed = config["phase2_reference_seed"] + fold_index * 100 + candidate_index
        state, record = _train_candidate(model_name, lr, config, candidate_seed, x_train, y_train, x_validation, y_validation, adjacency, batch_size)
        candidates.append(record)
        if selected is None or (record["best_source_validation_balanced_accuracy"], -candidate_index) > (selected["best_source_validation_balanced_accuracy"], -selected["candidate_index"]):
            selected_state = state
            selected = {**record, "candidate_index": candidate_index}
        print(f"{model_name} target={fold.held_out_subject} lr={lr:g} best_epoch={record['best_epoch']} source_val_bacc={record['best_source_validation_balanced_accuracy']:.4f}", flush=True)
    assert selected is not None and selected_state is not None
    model = build_model(model_name, config["dropout"], adjacency).to(device)
    model.load_state_dict(selected_state)
    train_probabilities, _ = _infer(model, x_train, batch_size)
    validation_probabilities, _ = _infer(model, x_validation, batch_size)
    train_metrics = _metrics(batch.y[train], train_probabilities.argmax(axis=1))
    validation_metrics = _metrics(y_validation, validation_probabilities.argmax(axis=1))
    probe = _probe_source_subjects(model, x_train, train, batch, fold, batch_size)
    # Target data are first read only after the source-only checkpoint has been selected.
    if model_name in RAW_MODELS:
        x_target = _tensor_from_raw(raw, target, mean, std, device)
    else:
        x_target = _tensor_from_de(batch, target, scaler, model_name, device)
    probabilities, embeddings = _infer(model, x_target, batch_size, embeddings=True)
    assert embeddings is not None and embeddings.shape == (len(target), 128)
    prediction = probabilities.argmax(axis=1)
    target_metrics = _metrics(batch.y[target], prediction)
    frame = oof_frame(batch, window_ids, np.asarray(batch.subject_ids, dtype=str) == fold.held_out_subject, prediction, probabilities, "class_probability", "raw_eeg" if model_name in RAW_MODELS else "raw_de", model_name, _session_numbers(batch))
    fold_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(fold_dir / "target_predictions.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        fold_dir / "target_embeddings.npz",
        dataset=np.asarray(["SEED"] * len(target)),
        subject_id=np.asarray(batch.subject_ids[target], dtype=str),
        session_id=np.asarray(batch.session_ids[target], dtype=str),
        trial_id=np.asarray(batch.trial_ids[target], dtype=str),
        window_id=window_ids[target],
        true_label=np.asarray(batch.y[target], dtype=np.int64),
        predicted_label=prediction,
        class_probabilities=probabilities.astype(np.float32),
        embedding=embeddings.astype(np.float32),
    )
    torch.save({"state_dict": selected_state, "fold": fold.to_dict(), "model": model_name, "selected": selected, "config_digest": digest}, fold_dir / "selected_checkpoint.pt")
    diagnostics = {
        "dataset": "SEED", "model": model_name, "fold": fold.to_dict(), "config_digest": digest,
        "seed": selected["seed"], "device": str(device), "selected_learning_rate": selected["learning_rate"],
        "best_epoch": selected["best_epoch"], "candidate_search": candidates,
        "checkpoint": str((fold_dir / "selected_checkpoint.pt").resolve()),
        "checkpoint_metric": config["checkpoint_metric"], "preprocessing": preprocessing,
        "graph": graph[1] if model_name == "graph_encoder" and graph is not None else None,
        "source_train": train_metrics, "source_validation": validation_metrics,
        "target": target_metrics, "subject_probe": probe,
        "training_instability": False,
    }
    write_json(diagnostic_path, diagnostics)
    print(f"{model_name} target={fold.held_out_subject} train={train_metrics['balanced_accuracy']:.4f} val={validation_metrics['balanced_accuracy']:.4f} target={target_metrics['balanced_accuracy']:.4f}", flush=True)
    del x_train, x_validation, x_target, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return diagnostics


def run_model(
    dataset: SEEDDataset, batch: FeatureBatch, window_ids: np.ndarray, model_name: str,
    config: dict[str, Any], output_root: Path, phase2_root: Path,
    target_subjects: set[str] | None = None, device_name: str | None = None,
) -> None:
    folds = phase2_folds(phase2_root, batch.subjects, config["phase2_reference_seed"])
    requested = batch.subjects if target_subjects is None else target_subjects
    if not requested or not requested <= batch.subjects:
        raise ValueError("Requested target subject IDs are invalid")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = raw_window_cache(dataset, batch, output_root / "cache") if model_name in RAW_MODELS else None
    graph = anatomical_adjacency(dataset.channel_names, config["graph_neighbors"]) if model_name == "graph_encoder" else None
    method_dir = output_root / model_name
    write_json(method_dir / "run_config.json", {**config, "model": model_name, "config_digest": config_digest(config), "device": str(device), "phase2_root": str(phase2_root.resolve())})
    for fold_index, fold in enumerate(folds):
        if fold.held_out_subject in requested:
            run_fold(dataset, batch, window_ids, fold, fold_index, model_name, config, output_root, phase2_root, raw, graph, device)


def finalize_model(batch: FeatureBatch, window_ids: np.ndarray, model_name: str, config: dict[str, Any], output_root: Path, phase2_root: Path) -> dict[str, Any]:
    folds = phase2_folds(phase2_root, batch.subjects, config["phase2_reference_seed"])
    method_dir = output_root / model_name
    parts: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = method_dir / "folds" / f"subject_{fold.held_out_subject}"
        info = json.loads((fold_dir / "diagnostics.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(fold_dir / "target_predictions.csv.gz", dtype={"subject_id": str, "session_id": str, "trial_id": str})
        validate_class_probabilities(frame)
        embedding_path = fold_dir / "target_embeddings.npz"
        with np.load(embedding_path, allow_pickle=False) as saved:
            if saved["embedding"].shape != (len(frame), 128) or not np.array_equal(saved["window_id"], frame["window_id"].to_numpy()):
                raise AssertionError(f"Embedding/OOF alignment mismatch in subject {fold.held_out_subject}")
            if not np.isfinite(saved["embedding"]).all():
                raise AssertionError("Nonfinite target embeddings")
            embedding = np.asarray(saved["embedding"], dtype=np.float32)
            needs_provenance = "class_probabilities" not in saved.files
            try:
                saved_trials = saved["trial_id"]
            except ValueError:
                # An earlier local export stored string keys as object arrays.
                # The float32 embeddings and integer window IDs remain readable without pickle.
                needs_provenance = True
            else:
                if not np.array_equal(saved_trials, frame["trial_id"].to_numpy()):
                    raise AssertionError(f"Embedding/OOF trial identity mismatch in subject {fold.held_out_subject}")
        if needs_provenance:
            # Earlier fold writers stored the same embedding and key rows separately.
            # Complete their export from the verified, aligned target prediction file.
            np.savez_compressed(
                embedding_path,
                dataset=np.asarray(frame["dataset"], dtype=str),
                subject_id=np.asarray(frame["subject_id"], dtype=str),
                session_id=np.asarray(frame["session_id"], dtype=str),
                trial_id=np.asarray(frame["trial_id"], dtype=str),
                window_id=frame["window_id"].to_numpy(),
                true_label=frame["true_label"].to_numpy(),
                predicted_label=frame["predicted_label"].to_numpy(),
                class_probabilities=frame[["score_negative", "score_neutral", "score_positive"]].to_numpy(dtype=np.float32),
                embedding=embedding,
            )
        with np.load(embedding_path, allow_pickle=False) as saved:
            for field in ("dataset", "subject_id", "session_id", "trial_id"):
                if not np.array_equal(saved[field].astype(str), frame[field].astype(str).to_numpy()):
                    raise AssertionError(f"Embedding export {field} differs from OOF")
            if not np.array_equal(saved["true_label"], frame["true_label"].to_numpy()) or not np.array_equal(saved["predicted_label"], frame["predicted_label"].to_numpy()):
                raise AssertionError("Embedding export labels or predictions differ from OOF")
            if not np.allclose(saved["class_probabilities"], frame[["score_negative", "score_neutral", "score_positive"]].to_numpy(), atol=1e-6):
                raise AssertionError("Embedding export probabilities differ from OOF")
        if info["fold"] != fold.to_dict() or info["config_digest"] != config_digest(config) or not Path(info["checkpoint"]).exists():
            raise AssertionError("Phase-3 fold config, reference split, or checkpoint mismatch")
        checkpoint = torch.load(info["checkpoint"], map_location="cpu", weights_only=True)
        if (checkpoint["fold"] != fold.to_dict() or checkpoint["model"] != model_name
                or checkpoint["config_digest"] != config_digest(config)
                or checkpoint["selected"]["best_epoch"] != info["best_epoch"]
                or checkpoint["selected"]["learning_rate"] != info["selected_learning_rate"]):
            raise AssertionError("Selected checkpoint differs from fold diagnostics")
        if set(info["preprocessing"]["fit_subjects"]) != set(fold.source_train_subjects):
            raise AssertionError("Preprocessing used non-training subjects")
        if set(frame["subject_id"].astype(str)) != {fold.held_out_subject}:
            raise AssertionError("Target predictions contain wrong subject")
        metrics = _metrics(frame["true_label"].to_numpy(), frame["predicted_label"].to_numpy())
        if metrics != info["target"]:
            raise AssertionError("Saved target metrics differ from saved predictions")
        subject_rows.append({"subject": fold.held_out_subject, "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"], **{f"F1_{name}": metrics["per_class_f1"][str(index)] for index, name in enumerate(CLASS_NAMES)}, "best_epoch": info["best_epoch"], "learning_rate": info["selected_learning_rate"], "n_windows": metrics["n_windows"]})
        for session_number in (1, 2, 3):
            session = frame.loc[frame["session_number"] == session_number]
            item = _metrics(session["true_label"].to_numpy(), session["predicted_label"].to_numpy())
            session_rows.append({"subject": fold.held_out_subject, "session": session_number, "accuracy": item["accuracy"], "balanced_accuracy": item["balanced_accuracy"], "macro_f1": item["macro_f1"], "n_windows": len(session)})
        probe_rows.append({"subject": fold.held_out_subject, **info["subject_probe"]})
        diagnostics.append(info)
        parts.append(frame)
    oof = pd.concat(parts, ignore_index=True)
    validate_oof(oof, batch, window_ids, tuple(sorted(batch.subjects, key=int)))
    reference_keys = ["dataset", "subject_id", "session_id", "trial_id", "window_id"]
    phase2_oof = pd.read_csv(
        phase2_root / "raw_de_logistic_regression" / "oof_predictions.csv.gz",
        usecols=reference_keys + ["true_label"],
        dtype={"subject_id": str, "session_id": str, "trial_id": str},
    )
    matched = phase2_oof.merge(oof[reference_keys + ["true_label"]], on=reference_keys, how="outer", indicator=True, validate="one_to_one", suffixes=("_phase2", "_phase3"))
    if (len(matched) != len(oof) or (matched["_merge"] != "both").any()
            or not np.array_equal(matched["true_label_phase2"], matched["true_label_phase3"])):
        raise AssertionError("Phase-3 target window keys or labels differ from canonical Phase-2 OOF")
    oof.to_csv(method_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(subject_rows).to_csv(method_dir / "subject_metrics.csv", index=False)
    pd.DataFrame(session_rows).to_csv(method_dir / "session_metrics.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(method_dir / "subject_probe.csv", index=False)
    summary = {
        "dataset": "SEED", "model": model_name, "n_oof_windows": len(oof),
        "statistical_unit": "held_out_subject", "aggregate": aggregate_subject_rows(subject_rows),
        "oof_confusion_matrix": confusion_matrix(oof["true_label"], oof["predicted_label"], labels=CLASSES).tolist(),
        "session_diagnostics": {str(session): {metric: float(np.mean([row[metric] for row in session_rows if row["session"] == session])) for metric in ("accuracy", "balanced_accuracy", "macro_f1")} for session in (1, 2, 3)},
        "training_validation_target_balanced_accuracy": {partition: float(np.mean([item[partition]["balanced_accuracy"] for item in diagnostics])) for partition in ("source_train", "source_validation", "target")},
        "subject_probe": {"mean_accuracy": float(np.mean([row["accuracy"] for row in probe_rows])), "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in probe_rows])), "chance": 1 / 11},
        "unstable_folds": [item["fold"]["held_out_subject"] for item in diagnostics if item["training_instability"]],
        "config_digest": config_digest(config),
    }
    write_json(method_dir / "summary.json", summary)
    manifest_path = method_dir / "run_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "architecture": ARCHITECTURE_DESCRIPTIONS[model_name],
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in build_model(model_name, config["dropout"], np.eye(62, dtype=np.float32) if model_name == "graph_encoder" else None).parameters())),
        "architecture_source": str((Path(__file__).parent / "phase3_models.py").resolve()),
        "training_source": str(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "candidate_seed_rule": "20260928 + 100 * zero_based_phase2_fold_index + zero_based_learning_rate_candidate_index",
        "determinism_controls": "Python, NumPy, CPU and CUDA seeds; deterministic cuDNN; torch deterministic algorithms with warnings; CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "checkpoint_selection_scope": "three source-validation subjects only; target read after checkpoint selection",
        "candidate_history_location": "folds/subject_<ID>/diagnostics.json",
    })
    write_json(manifest_path, manifest)
    return summary
