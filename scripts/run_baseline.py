from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strict_eeg_benchmark.artifacts import write_json, write_predictions
from src.strict_eeg_benchmark.datasets import SEEDDataset
from src.strict_eeg_benchmark.evaluation import aggregate_subject_metrics, classification_metrics
from src.strict_eeg_benchmark.preprocessing import AuditedPCA, AuditedSelectKBest, AuditedStandardScaler
from src.strict_eeg_benchmark.splits import make_loso_folds
from src.strict_eeg_benchmark.training import select_model
from src.strict_eeg_benchmark.types import FeatureBatch, Partition


def transformed(batch: FeatureBatch, X: np.ndarray) -> FeatureBatch:
    return FeatureBatch(
        X=X,
        y=batch.y,
        subject_ids=batch.subject_ids,
        session_ids=batch.session_ids,
        trial_ids=batch.trial_ids,
        dataset_name=batch.dataset_name,
        feature_names=batch.feature_names,
        partition=batch.partition,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strict source-only LOSO EEG baseline")
    parser.add_argument("--dataset", choices=["seed"], required=True)
    parser.add_argument("--model", choices=["logistic_regression", "svm"], required=True)
    parser.add_argument("--features", choices=["de"], required=True)
    parser.add_argument("--protocol", choices=["loso-dg"], required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_seed.yaml")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "SEED")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "benchmark_runs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true", help="Run one fold with two windows per trial")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    max_windows = 2 if args.smoke_test else config["data"]["max_windows_per_trial"]
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_seed_{args.model}_{seed}"
    run_dir = args.output_root / experiment_id
    effective_config = {
        **config,
        "experiment_id": experiment_id,
        "dataset": args.dataset,
        "model_name": args.model,
        "features_name": args.features,
        "protocol": args.protocol,
        "seed": seed,
        "smoke_test": args.smoke_test,
        "effective_max_windows_per_trial": max_windows,
        "data_root": str(args.data_root.resolve()),
    }
    write_json(run_dir / "config.json", effective_config)

    dataset = SEEDDataset(args.data_root)
    data = dataset.load_de_features(max_windows_per_trial=max_windows)
    folds = make_loso_folds(dataset.subject_ids, seed, float(config["experiment"]["validation_fraction"]))
    if args.smoke_test:
        folds = folds[:1]
    subject_rows: list[dict[str, object]] = []

    for fold in folds:
        train = data.subset_subjects(set(fold.source_train_subjects), Partition.SOURCE_TRAIN)
        validation = data.subset_subjects(set(fold.source_validation_subjects), Partition.SOURCE_VALIDATION)
        target = data.subset_subjects({fold.held_out_subject}, Partition.TARGET_TEST)

        transformers = []
        if config["preprocessing"]["standard_scaler"]:
            transformers.append(AuditedStandardScaler())
        if config["preprocessing"]["pca"] is not None:
            transformers.append(AuditedPCA(config["preprocessing"]["pca"], random_state=seed))
        if config["preprocessing"]["feature_selection_k"] is not None:
            transformers.append(AuditedSelectKBest(config["preprocessing"]["feature_selection_k"]))

        train_transformed = train
        validation_transformed = validation
        target_transformed = target
        for transformer in transformers:
            transformer.fit(train_transformed, fold)
            train_transformed = transformed(train_transformed, transformer.transform(train_transformed.X))
            validation_transformed = transformed(
                validation_transformed, transformer.transform(validation_transformed.X)
            )
            target_transformed = transformed(target_transformed, transformer.transform(target_transformed.X))

        selection = select_model(
            args.model,
            train_transformed,
            validation_transformed,
            fold,
            [float(value) for value in config["model"]["candidate_C"]],
            seed,
        )
        predictions = selection.model.predict(target_transformed.X)
        metrics = classification_metrics(target.y, predictions, labels=(0, 1, 2))
        metrics.update({"held_out_subject": fold.held_out_subject})
        subject_rows.append(metrics)

        fold_dir = run_dir / "folds" / f"subject_{fold.held_out_subject}"
        write_json(
            fold_dir / "metadata.json",
            {
                "experiment_id": experiment_id,
                "dataset": dataset.name,
                **fold.to_dict(),
                "selected_C": selection.selected_C,
                "validation_scores": list(selection.validation_scores),
                "model_fit_subjects": list(selection.fit_subjects),
                "feature_source": config["data"]["feature_source"],
            },
        )
        write_json(fold_dir / "metrics.json", metrics)
        write_json(fold_dir / "preprocessing.json", [transformer.metadata() for transformer in transformers])
        write_predictions(
            fold_dir / "predictions.npz",
            predictions,
            target.y,
            target.subject_ids,
            target.session_ids,
            target.trial_ids,
        )
        print(
            f"target={fold.held_out_subject} accuracy={metrics['accuracy']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )

    summary = {
        "experiment_id": experiment_id,
        "dataset": dataset.name,
        "protocol": args.protocol,
        "model": args.model,
        "subject_metrics": subject_rows,
        "aggregate": aggregate_subject_metrics(subject_rows),
        "warning": "Smoke-test summaries cover only the explicitly run held-out subjects." if args.smoke_test else None,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
