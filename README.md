# Cross-subject EEG emotion recognition: evaluation protocols

Code and fixed experiment specifications for a study of evaluation protocols in cross-subject EEG emotion recognition on SEED, SEED-IV, FACED, and DEAP. The repository supports reproduction of source-subject allocation, checkpoint selection, target-data access, preprocessing, and subject-level evaluation. Licensed datasets and previously generated results are not distributed.

## Repository contents

- `src/`: dataset interfaces, split/leakage guards, preprocessing, model implementations, and fold-level training and validation code.
- `scripts/`: dataset preparation, fold-level training entry points, analysis, and offline verification.
- `configs/` and `folds/`: fixed settings and subject assignments. The DEAP source-validation assignments predate the workbook-label reconciliation and were retained unchanged.
- `experiments/`: fixed replication manifests and configuration snapshots required by the runners.

The repository excludes raw EEG, provider feature arrays, the licensed DEAP ratings workbook, derived trial-level DEAP labels, model checkpoints, training trajectories, results, metadata, and manuscript files. Dataset access remains subject to the providers' terms. The repository does not currently grant a code license.

## Offline verification

Python 3.13 was used for the current checks. Create an environment and install the requirements appropriate to the intended task. The lightweight verification path needs NumPy, SciPy, scikit-learn, pandas, PyYAML, and pytest; model training additionally needs PyTorch and PyTorch Geometric.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m pytest tests -q
```

The offline checks validate the fixed folds and leakage guards without accessing any EEG recordings. Full training and result reconstruction require separately licensed data and the intermediate caches described below.

## Training from licensed data

The model runners use the fixed configurations in `configs/` and the frozen fold files. Place licensed datasets and documented local caches at the relative paths in [Data access and preparation](docs/DATA_ACCESS.md). Install the pinned LibEER dependency as described in [Third-party components](docs/THIRD_PARTY.md). Example fold commands:

```bash
python scripts/run_p3_seed.py --model DGCNN --setting strict_11_3_1 --target 1
python scripts/run_p3_ta_u.py --target 1
python scripts/run_p4_provider.py --dataset SEED-IV --model DGCNN --target 1
python scripts/run_p4_deap.py --task valence --target 01
```

Source-only configurations fit preprocessing and select checkpoints using source subjects. The DANN comparator is separately labelled target-unlabelled adaptation; it uses held-out target EEG without target emotion labels. Retrospective target-selected checkpoint results are diagnostic and are not deployable source-only estimates.

Run outputs are written locally and ignored by Git. The original manuscript results cannot be independently regenerated from this repository alone because licensed inputs and saved trajectories are not redistributed; rerunning the fixed configurations on those inputs is required.
