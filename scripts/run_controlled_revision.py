"""Run fixed-budget checkpoint, source-training, and adaptation controls.

Licensed feature caches and archived fold definitions stay local. All outputs
are written to --output-dir; no target emotion labels enter model fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics(labels: np.ndarray, probability: np.ndarray, classes: int) -> dict:
    predicted = probability.argmax(1)
    matrix = np.bincount(labels * classes + predicted, minlength=classes ** 2).reshape(classes, classes)
    support = matrix.sum(1)
    if np.any(support == 0):
        raise AssertionError("A declared class is absent from a scoring person")
    recall = np.diag(matrix) / support
    denom = matrix.sum(0) + support
    f1 = np.divide(2 * np.diag(matrix), denom, out=np.zeros(classes), where=denom != 0)
    return {"balanced_accuracy": float(recall.mean()),
            "accuracy": float(np.trace(matrix) / matrix.sum()),
            "macro_f1": float(f1.mean()), "class_support": support.tolist()}


class BatchCycle:
    def __init__(self, n: int, size: int, seed: int):
        self.n, self.size = n, size
        self.rng = np.random.default_rng(seed)
        self.remaining = np.empty(0, dtype=np.int64)

    def next(self) -> np.ndarray:
        while len(self.remaining) < self.size:
            self.remaining = np.concatenate([self.remaining, self.rng.permutation(self.n)])
        batch, self.remaining = self.remaining[:self.size], self.remaining[self.size:]
        return batch


def load_panel(project: Path, data_parent: Path, dataset: str):
    from src.tac_revision.provider_training import Panel, load_panel as provider
    if dataset == "SEED":
        from src.p3_protocol_benchmark.data import load_seed_lds
        source = load_seed_lds(data_parent)
        take = np.flatnonzero(source.session_ordinal == 1)
        trial = source.trial[take]
        local = np.empty(len(take), dtype=np.int32)
        for name in dict.fromkeys(trial):
            rows = np.flatnonzero(trial == name)
            local[rows] = np.arange(len(rows))
        return Panel(source.x[take], source.y[take], source.subject[take], trial,
                     local, 62, 3, "SEED provider DE-LDS, first session only")
    if dataset == "FACED":
        from analyze_cross_person_checkpoint import load_faced_panel
        return load_faced_panel(project, data_parent)
    return provider(project, dataset)


def archived_fold(project: Path, dataset: str, target: str) -> dict:
    if dataset == "SEED":
        path = project / "experiments/p3_seed/runs/strict_11_3_1_session1/DGCNN/none" / f"target_{target}/diagnostics.json"
    else:
        path = project / "experiments/p4_cross_dataset/runs" / dataset.replace("-", "") / "DGCNN/strict" / f"target_{target}/diagnostics.json"
    if not path.exists():
        manifest = Path(__file__).resolve().parents[1] / "folds/FOLD_MANIFEST.csv"
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            matches = [r for r in csv.DictReader(handle)
                       if r["dataset"] == dataset and r["task"] == "emotion"
                       and r["target_subject"] == target]
        if len(matches) != 1:
            raise AssertionError(f"Expected one public fold specification for {dataset} {target}")
        row = matches[0]
        train = json.loads(row["source_train_subjects"])
        validation = json.loads(row["source_validation_subjects"])
        if set(train) & set(validation) or target in set(train) | set(validation):
            raise AssertionError("Public fold specification crosses subject boundaries")
        return {"train": train, "validation": validation, "target": target,
                "archive": str(manifest), "archive_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    row = json.loads(path.read_text(encoding="utf-8"))
    train = row.get("train_subjects", row.get("source_train_subjects"))
    validation = row.get("validation_subjects", row.get("source_validation_subjects"))
    actual_target = row.get("target_subjects", [row.get("target_subject")])
    if actual_target != [target] or set(train) & set(validation) or target in set(train) | set(validation):
        raise AssertionError(f"Invalid archived partition: {path}")
    return {"train": train, "validation": validation, "target": target,
            "archive": str(path), "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def anchored(panel, people: list[str], dataset: str) -> np.ndarray:
    mask = np.isin(panel.subject, people)
    if dataset == "FACED":
        mask &= np.isin(panel.local_window, np.linspace(0, 29, 10, dtype=int))
    else:
        mask &= panel.local_window % 4 == 0
    return np.flatnonzero(mask)


def construct_model(channels: int, classes: int, seed: int, domain: bool, config: dict):
    from models.DGCNN import DGCNN, laplacian
    from models.DannDgcnn import Discriminator, ReverseLayerF

    class LockedDGCNN(DGCNN):
        def get_param(self):
            pass

    class ControlledDomainGraph(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.discriminator = Discriminator(channels * encoder.layers[-1], 256, 2)

        def features(self, x):
            e = self.encoder
            lap = laplacian(e.relu(e.adj + e.adj_bias))
            for conv, activation in zip(e.graphConvs, e.b_relus):
                x = activation(e.dropout(conv(x, lap)))
            return x.reshape(len(x), -1)

        def classify(self, features):
            e = self.encoder
            return e.fc2(e.dropout(e.fc(e.dropout(features))))

        def forward(self, x):
            return self.classify(self.features(x))

        def domain_logits(self, features):
            reversed_features = ReverseLayerF.apply(features, config["gradient_reversal_alpha"])
            return self.discriminator(reversed_features)

    torch.manual_seed(seed)
    encoder = LockedDGCNN(channels, 5, classes, k=2, relu_is=1, dropout_rate=.5)
    encoder_hash = state_digest(encoder.state_dict())
    if domain:
        model = ControlledDomainGraph(encoder)
    else:
        model = encoder
    return model.to("cuda"), encoder_hash


@torch.no_grad()
def predict(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(x), 1024):
        values = torch.as_tensor(np.array(x[start:start + 1024], dtype=np.float32), device="cuda")
        parts.append(torch.softmax(model(values), 1).cpu().numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-5):
        raise AssertionError("Invalid predicted probability vectors")
    return result


def fit_path(model, x: np.ndarray, y: np.ndarray, total_steps: int,
             checkpoints: list[int], seed: int, config: dict,
             auxiliary: np.ndarray | None = None, domain_weight: float = 0.0):
    started = time.perf_counter()
    torch.manual_seed(seed + 99)
    torch.cuda.manual_seed_all(seed + 99)
    tx = torch.as_tensor(np.array(x, dtype=np.float32), device="cuda")
    ty = torch.as_tensor(np.array(y, dtype=np.int64), device="cuda")
    aux = (torch.as_tensor(np.array(auxiliary, dtype=np.float32), device="cuda")
           if auxiliary is not None else None)
    source_cycle = BatchCycle(len(x), config["batch_size"], seed + 17)
    auxiliary_rng = np.random.default_rng(seed + 31)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"], eps=config["optimizer_eps"])
    snapshots, history = {}, []
    loss_sum = 0.0
    model.train()
    for step in range(1, total_steps + 1):
        index = torch.as_tensor(source_cycle.next(), device="cuda")
        xb, yb = tx[index], ty[index]
        optimizer.zero_grad(set_to_none=True)
        if aux is None:
            logits = model(xb)
            parameters = model.parameters()
        else:
            fs = model.features(xb)
            logits = model.classify(fs)
            ai = torch.as_tensor(auxiliary_rng.integers(len(aux), size=len(xb)), device="cuda")
            fu = model.features(aux[ai])
            domain_logits = model.domain_logits(torch.cat([fs, fu]))
            domain_y = torch.cat([torch.zeros(len(xb), device="cuda", dtype=torch.long),
                                  torch.ones(len(xb), device="cuda", dtype=torch.long)])
            parameters = model.encoder.parameters()
        loss = nn.functional.cross_entropy(logits, yb)
        if aux is not None:
            loss = loss + domain_weight * nn.functional.cross_entropy(domain_logits, domain_y)
        loss = loss + config["graph_norm_penalty"] * sum(torch.norm(p) for p in parameters)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
        if step in checkpoints:
            snapshots[step] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append({"step": step, "mean_loss_so_far": loss_sum / step})
    if set(snapshots) != set(checkpoints):
        raise AssertionError("Incomplete checkpoint coverage")
    return snapshots, history, time.perf_counter() - started


def trial_metrics(panel, rows, probability):
    names = panel.trial[rows]
    labels, means = [], []
    for name in dict.fromkeys(names):
        take = names == name
        truth = np.unique(panel.y[rows][take])
        if len(truth) != 1:
            raise AssertionError("Trial truth is inconsistent")
        labels.append(truth[0])
        means.append(probability[take].mean(0))
    return metrics(np.array(labels), np.array(means), panel.n_classes)


def checkpoint_case(panel, fold, dataset, seed, config, folder, identity, pilot):
    rows = anchored(panel, fold["train"], dataset)
    steps = config["checkpoint_counts"][dataset] * math.ceil(len(rows) / config["batch_size"])
    count = config["checkpoint_counts"][dataset]
    if pilot:
        steps, count = 32, 2
    candidates = np.linspace(steps // count, steps, count, dtype=int).tolist()
    model, initial_hash = construct_model(panel.n_channels, panel.n_classes, seed, False, config)
    snapshots, history, wall = fit_path(model, panel.x[rows], panel.y[rows], steps, candidates, seed, config)
    val_people = fold["validation"]
    val_rows = np.flatnonzero(np.isin(panel.subject, val_people))
    val_subject = panel.subject[val_rows]
    val_y = panel.y[val_rows]
    scores = np.empty((len(val_people), count))
    for j, step in enumerate(candidates):
        model.load_state_dict(snapshots[step])
        probability = predict(model, panel.x[val_rows])
        for i, person in enumerate(val_people):
            take = val_subject == person
            scores[i, j] = metrics(val_y[take], probability[take], panel.n_classes)["balanced_accuracy"]
    chosen = scores.argmax(1).tolist()
    pooled = int(scores.mean(0).argmax())
    write_json(folder / "selection.json", {"candidate_steps": candidates,
        "validation_people": val_people, "individual_candidate_indices": chosen,
        "mean_source_validation_candidate_index": pooled,
        "target_emotion_labels_used": False})
    target_rows = np.flatnonzero(panel.subject == fold["target"])
    probabilities = []
    target_curve = []
    for step in candidates:
        model.load_state_dict(snapshots[step])
        probability = predict(model, panel.x[target_rows])
        probabilities.append(probability)
        target_curve.append(metrics(panel.y[target_rows], probability, panel.n_classes)["balanced_accuracy"])
        path = folder / "trajectory" / f"step_{step:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshots[step], path)
    rotations = [{"selector_person": person, "selected_step": candidates[j],
        "selector_gain_vs_final_pp": 100 * float(scores[i, j] - scores[i, -1]),
        "target_gain_vs_final_pp": 100 * float(target_curve[j] - target_curve[-1])}
        for i, (person, j) in enumerate(zip(val_people, chosen))]
    np.savez_compressed(folder / "scoring_arrays.npz", validation_bacc=scores,
        target_probabilities=np.stack(probabilities), target_labels=panel.y[target_rows],
        target_trial=panel.trial[target_rows], source_rows=rows)
    result = {"identity": identity, "status": "complete", "dataset": dataset,
        "target": fold["target"], "seed": seed, "arm": "strict",
        "fold": fold, "initial_encoder_sha256": initial_hash,
        "training_examples": len(rows), "training_people": len(fold["train"]),
        "class_counts": np.bincount(panel.y[rows], minlength=panel.n_classes).tolist(),
        "updates": steps, "examples_processed": steps * config["batch_size"],
        "candidate_steps": candidates, "target_bacc_curve": target_curve,
        "rotations": rotations, "mean_selector_gain_vs_final_pp": float(np.mean([r["selector_gain_vs_final_pp"] for r in rotations])),
        "mean_target_gain_vs_final_pp": float(np.mean([r["target_gain_vs_final_pp"] for r in rotations])),
        "source_selected_target_bacc": target_curve[pooled],
        "final_window": metrics(panel.y[target_rows], probabilities[-1], panel.n_classes),
        "final_trial": trial_metrics(panel, target_rows, probabilities[-1]),
        "training_seconds": wall, "history": history}
    write_json(folder / "result.json", result)
    return result


def source_pools(panel, fold, dataset, seed):
    strict = anchored(panel, fold["train"], dataset)
    people = sorted(set(fold["train"]) | set(fold["validation"]), key=int)
    eligible = anchored(panel, people, dataset)
    counts = np.bincount(panel.y[strict], minlength=panel.n_classes)
    available = np.bincount(panel.y[eligible], minlength=panel.n_classes)
    proportions = counts / counts.sum()
    total = int(np.floor(np.min(available / proportions)))
    expanded_counts = np.floor(total * proportions).astype(int)
    for c in np.argsort(-(total * proportions - expanded_counts))[:total - expanded_counts.sum()]:
        expanded_counts[c] += 1
    if np.any(expanded_counts > available) or np.any(expanded_counts < counts):
        raise AssertionError("Source class-matched expansion infeasible")
    rng = np.random.default_rng(seed + 71)
    matched, expanded = [], []
    for c in range(panel.n_classes):
        shuffled = rng.permutation(eligible[panel.y[eligible] == c])
        expanded.extend(shuffled[:expanded_counts[c]])
        matched.extend(shuffled[:counts[c]])
    matched, expanded = np.sort(matched), np.sort(expanded)
    if set(panel.subject[matched]) != set(people) or set(panel.subject[expanded]) != set(people):
        raise AssertionError("Broader source pool does not represent all declared people")
    if not set(matched) <= set(expanded) or fold["target"] in set(panel.subject[expanded]):
        raise AssertionError("Source subset nesting or target exclusion failed")
    return strict, matched, expanded, people


def source_case(panel, fold, dataset, seed, config, folder, identity, pilot):
    strict, matched, expanded, people = source_pools(panel, fold, dataset, seed)
    s = config["checkpoint_counts"][dataset] * math.ceil(len(strict) / config["batch_size"])
    extended = config["checkpoint_counts"][dataset] * math.ceil(len(expanded) / config["batch_size"])
    if pilot:
        s, extended = 32, 40
    targets = np.flatnonzero(panel.subject == fold["target"])
    output = []
    for name, rows, steps, checkpoints in [
        ("broader_matched", matched, s, [s]),
        ("broader_expanded", expanded, extended, [s, extended]),
    ]:
        model, initial_hash = construct_model(panel.n_channels, panel.n_classes, seed, False, config)
        snapshots, history, wall = fit_path(model, panel.x[rows], panel.y[rows], steps, checkpoints, seed, config)
        scores = {}
        for step in checkpoints:
            model.load_state_dict(snapshots[step])
            probability = predict(model, panel.x[targets])
            scores[str(step)] = {"window": metrics(panel.y[targets], probability, panel.n_classes),
                                "trial": trial_metrics(panel, targets, probability)}
            path = folder / name / f"step_{step:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(snapshots[step], path)
        np.save(folder / name / "source_rows.npy", rows)
        output.append({"arm": name, "training_examples": len(rows), "training_people": len(people),
            "class_counts": np.bincount(panel.y[rows], minlength=panel.n_classes).tolist(),
            "initial_encoder_sha256": initial_hash, "updates": checkpoints,
            "examples_processed": [step * config["batch_size"] for step in checkpoints],
            "scores": scores, "training_seconds": wall, "history": history})
    write_json(folder / "result.json", {"identity": identity, "status": "complete",
        "dataset": dataset, "target": fold["target"], "seed": seed,
        "strict_steps": s, "extended_steps": extended, "arms": output, "fold": fold})


def adaptation_case(panel, fold, dataset, seed, config, folder, identity, pilot):
    source = anchored(panel, fold["train"], dataset)
    auxiliary_rows = anchored(panel, [fold["target"]], dataset)
    target = np.flatnonzero(panel.subject == fold["target"])
    rng = np.random.default_rng(seed + 83)
    proxy = rng.choice(source, size=len(auxiliary_rows), replace=False)
    s = config["checkpoint_counts"][dataset] * math.ceil(len(source) / config["batch_size"])
    if pilot:
        s = 32
    arms = [("loss_off_proxy", proxy, 0.0),
            ("loss_on_proxy", proxy, config["domain_loss_weight"]),
            ("loss_on_target", auxiliary_rows, config["domain_loss_weight"])]
    if pilot:
        arms.append(("loss_off_target_negative_control", auxiliary_rows, 0.0))
    output, zero_hash = [], None
    for name, auxiliary, weight in arms:
        model, initial_hash = construct_model(panel.n_channels, panel.n_classes, seed, True, config)
        snapshots, history, wall = fit_path(model, panel.x[source], panel.y[source], s, [s], seed,
                                          config, auxiliary=panel.x[auxiliary], domain_weight=weight)
        encoder_hash = state_digest(model.encoder.state_dict())
        if name == "loss_off_proxy":
            zero_hash = encoder_hash
        if name == "loss_off_target_negative_control" and encoder_hash != zero_hash:
            raise AssertionError("Loss-off target features changed encoder/classifier parameters")
        probability = predict(model, panel.x[target])
        destination = folder / name
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(snapshots[s], destination / "final.pt")
        np.save(destination / "auxiliary_rows.npy", auxiliary)
        output.append({"arm": name, "source_examples": len(source), "auxiliary_examples": len(auxiliary),
            "source_people": fold["train"], "auxiliary_people": sorted(set(panel.subject[auxiliary]), key=int),
            "source_updates": s, "auxiliary_forward_batches": s, "examples_per_batch": config["batch_size"],
            "domain_loss_weight": weight, "gradient_reversal_alpha": config["gradient_reversal_alpha"],
            "initial_encoder_sha256": initial_hash, "final_encoder_sha256": encoder_hash,
            "window": metrics(panel.y[target], probability, panel.n_classes),
            "trial": trial_metrics(panel, target, probability),
            "target_labels_used_in_fitting": False, "training_seconds": wall, "history": history})
    write_json(folder / "result.json", {"identity": identity, "status": "complete", "dataset": dataset,
        "target": fold["target"], "seed": seed, "arms": output, "fold": fold,
        "loss_off_target_control_passed": zero_hash == output[-1]["final_encoder_sha256"] if pilot else None})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/controlled_protocol_revision.json")
    parser.add_argument("--stage", choices=["all", "checkpoint", "source", "adaptation"], default="all")
    parser.add_argument("--datasets", nargs="+", choices=["SEED", "SEED-IV", "FACED"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "tmp/p3_references/LibEER/LibEER"))
    if not torch.cuda.is_available():
        raise RuntimeError("This planned campaign requires a CUDA device")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text())
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output_dir.resolve() / ("pilot" if args.pilot else "production")
    output.mkdir(parents=True, exist_ok=True)
    environment_path = output / "run_environments" / f"process_{os.getpid()}_{int(time.time())}.json"
    write_json(environment_path, {"config": config, "script_sha256": code_hash,
        "torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name(0),
        "pilot": args.pilot, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    datasets = args.datasets or config["checkpoint_datasets"]
    seeds = args.seeds or config["training_seeds"]
    completed = 0
    for dataset in datasets:
        panel = load_panel(project, args.data_parent, dataset)
        targets = sorted(set(panel.subject), key=(None if dataset == "FACED" else int))
        if args.limit_targets:
            targets = targets[:args.limit_targets]
        for seed in seeds:
            for target in targets:
                fold = archived_fold(project, dataset, target)
                if set(fold["train"]) | set(fold["validation"]) | {target} != set(panel.subject):
                    raise AssertionError("Fold coverage changed")
                stages = [args.stage] if args.stage != "all" else ["checkpoint", "source", "adaptation"]
                for stage in stages:
                    if stage == "source" and dataset not in config["source_control_datasets"]:
                        continue
                    if stage == "adaptation" and dataset not in config["adaptation_control_datasets"]:
                        continue
                    folder = output / dataset.replace("-", "") / f"seed_{seed}" / f"target_{target}" / stage
                    identity = digest({"config": config, "code_hash": code_hash, "fold": fold,
                                       "seed": seed, "stage": stage, "pilot": args.pilot})
                    record = folder / "result.json"
                    if record.exists():
                        old = json.loads(record.read_text())
                        if old["identity"] != identity or old["status"] != "complete":
                            raise AssertionError(f"Existing run identity differs: {folder}")
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    began = time.perf_counter()
                    if stage == "checkpoint":
                        checkpoint_case(panel, fold, dataset, seed, config, folder, identity, args.pilot)
                    elif stage == "source":
                        source_case(panel, fold, dataset, seed, config, folder, identity, args.pilot)
                    else:
                        adaptation_case(panel, fold, dataset, seed, config, folder, identity, args.pilot)
                    completed += 1
                    print(f"DONE {dataset} seed={seed} target={target} stage={stage} "
                          f"seconds={time.perf_counter() - began:.1f} completed_this_run={completed}", flush=True)
    print("CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
