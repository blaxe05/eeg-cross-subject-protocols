# Cross-subject EEG emotion recognition: evaluation protocols

Code and fixed experiment specifications for a study of evaluation protocols in cross-subject EEG emotion recognition on SEED, SEED-IV, FACED, and DEAP. The repository supports reproduction of source-subject allocation, checkpoint selection, target-data access, preprocessing, and subject-level evaluation. Licensed datasets and previously generated results are not distributed.

## Repository contents

- `src/`: dataset interfaces, split/leakage guards, preprocessing, model implementations, and fold-level training and validation code.
- `scripts/`: dataset preparation, fold-level training entry points, analysis, and offline verification.
- `configs/` and `folds/`: fixed settings and subject assignments. The DEAP source-validation assignments predate the workbook-label reconciliation and were retained unchanged.
- `experiments/`: fixed replication manifests and configuration snapshots required by the runners.

The repository excludes raw EEG, provider feature arrays, the licensed DEAP ratings workbook, derived trial-level DEAP labels, model checkpoints, training trajectories, results, metadata, and manuscript files. Dataset access remains subject to the providers' terms.

## Reproducibility levels

### Offline verification

Python 3.13 was used for the current checks. Create an environment and install the requirements appropriate to the intended task. The lightweight verification path needs NumPy, SciPy, scikit-learn, pandas, PyYAML, and pytest; model training additionally needs PyTorch and PyTorch Geometric.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m pytest tests -q
```

The offline checks validate fixed folds and leakage guards without accessing EEG recordings. The repository contains the code and specifications, but no published result CSVs.

### Full retraining

The model runners use the fixed configurations in `configs/` and the frozen fold files. Place licensed datasets and documented local caches at the relative paths in [Data access and preparation](docs/DATA_ACCESS.md). Install the pinned LibEER dependency as described in [Third-party components](docs/THIRD_PARTY.md). Example fold commands:

```bash
python scripts/run_p3_seed.py --model DGCNN --setting strict_11_3_1 --target 1
python scripts/run_p3_ta_u.py --target 1
python scripts/run_p4_provider.py --dataset SEED-IV --model DGCNN --target 1
python scripts/run_p4_deap.py --task valence --target 01
```

Source-only configurations fit preprocessing and select checkpoints using source subjects. The DANN comparator is separately labelled target-unlabelled adaptation; it uses held-out target EEG without target emotion labels. Retrospective target-selected checkpoint results are diagnostic and are not deployable source-only estimates.

### Published-result reconstruction

Run outputs are written locally and ignored by Git. Exact manuscript numbers cannot be reconstructed from this repository alone: licensed datasets, local feature caches, and saved trajectories are not redistributed. Retraining requires the documented seeds and software environment; stochastic training can yield different numbers. The manuscript tables and figures are not part of this code repository.

### Figure rendering from local summaries

The code-only figure utilities are in `scripts/figures/`. The result plotter requires the locally generated `C1_EQUAL_SUBJECT_PRIMARY.csv`, `C1_CANDIDATE_BUDGET_SUMMARY.csv`, `C2_MATCHED_STATS.csv`, `C3_DANN_STATS.csv`, `PER_CLASS_CANONICAL.csv`, and `CANONICAL_RESULT_SUMMARY.csv` in one directory. These derived result files are not distributed here.

```bash
python scripts/figures/plot_result_figures.py --data-dir path/to/local/summaries --output-dir path/to/figures
python scripts/figures/draw_contrast_flow.py --output-prefix path/to/figures/fig_contrast_flow
```

The plotter reads the summary values without retraining models or recomputing metrics; it writes vector PDFs and PNG previews. The flow diagram has no data dependency.

## License

Original author-written code in this repository is released under the [MIT License](LICENSE). The CDCN evaluation wrapper in `src/tac_revision/models.py` adapts methods from LibEER; its upstream copyright and MIT terms are preserved in [third-party notices](docs/THIRD_PARTY.md). Third-party libraries, benchmark implementations, and datasets remain subject to their respective licenses and terms. This repository does not relicense or redistribute SEED, SEED-IV, FACED, DEAP, LibEER, or other third-party materials. Fold manifests are experiment specifications, not dataset redistribution.
