# Cross-subject EEG emotion recognition: evaluation protocols

Code and fixed experiment specifications for controlled evaluation studies in cross-subject EEG emotion recognition on SEED, SEED-IV, and FACED. The repository supports reproduction of source-subject allocation, checkpoint selection, target-data access, preprocessing, and subject-level evaluation. Historical DEAP utilities remain for provenance; DEAP quantitative results are excluded from the revised submission because local rating sources could not be independently authenticated. Licensed datasets and previously generated results are not distributed.

## Repository contents

- `src/`: dataset interfaces, split/leakage guards, preprocessing, model implementations, and fold-level training and validation code.
- `scripts/`: dataset preparation, fold-level training entry points, analysis, and offline verification.
- `configs/` and `folds/`: fixed settings and subject assignments. The DEAP source-validation assignments predate the workbook-label reconciliation and were retained unchanged.
- `experiments/`: fixed replication manifests and configuration snapshots required by the runners.

The repository excludes raw EEG, provider feature arrays, the licensed DEAP ratings workbook, derived trial-level DEAP labels, model checkpoints, training trajectories, results, metadata, and manuscript files. Dataset access remains subject to the providers' terms.

## Reproducibility levels

### Offline verification

Python 3.13 was used for the current checks. The controlled reruns use Python 3.13.11, NumPy 2.2.6, SciPy 1.16.0, scikit-learn 1.7.0, and PyTorch 2.7.1 with CUDA 12.8. Create an environment and install the requirements appropriate to the intended task. The lightweight verification path needs NumPy, SciPy, scikit-learn, pandas, PyYAML, and pytest; model training additionally needs PyTorch and PyTorch Geometric. Reading the local DEAP `.xls` audit workbook requires xlrd.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m pytest tests -q
```

The offline checks validate fixed folds and leakage guards without accessing EEG recordings. The repository contains the code and specifications, but no published result CSVs.

### Full retraining

The controlled experiment runner uses three fixed training seeds and separates checkpoint selection, source-person composition, available examples, optimizer exposure, and unlabelled target input under a fixed adaptation objective:

```bash
python scripts/run_controlled_revision.py --project-root . --data-parent /path/to/dataset-parent --output-dir /path/to/local/controls --pilot --limit-targets 1 --seeds 2031
python scripts/verify_controlled_revision.py --run-dir /path/to/local/controls/pilot
python scripts/run_controlled_revision.py --project-root . --data-parent /path/to/dataset-parent --output-dir /path/to/local/controls
python scripts/verify_controlled_revision.py --run-dir /path/to/local/controls/production
python scripts/analyze_controlled_revision.py --run-dir /path/to/local/controls/production --output-dir /path/to/local/controls/analysis
```

Settings are fixed in `configs/controlled_protocol_revision.json`. The runner accepts the public `folds/FOLD_MANIFEST.csv` when historical private diagnostics are absent. It uses fixed-size batches and update budgets; its checkpoint baseline is the prescribed final step. Source controls match examples/classes/updates before separately expanding the data and optimization exposure. The controlled adaptation wrapper retains the same DGCNN encoder/classifier in every arm and compares an equally sized source proxy with real unlabelled target EEG at the same domain-loss coefficient. A pilot loss-off negative control requires target and proxy inputs to produce identical encoder/classifier parameters.

See [Controlled experiment specification](docs/CONTROLLED_EXPERIMENTS.md) for the paired comparisons, complete 639-record scope, and interpretation limits.

### Architecture and source-subset replication

The fixed extension adds a CDCN replication on the same three datasets and three seeds, plus two additional source-subset draws for both backbones on SEED/SEED-IV. It reuses the completed DGCNN experiment without modifying its outputs. All new artifacts remain local.

```bash
python scripts/run_architecture_replication.py --project-root . --data-parent /path/to/dataset-parent --output-dir /path/to/local/replication --previous-run-dir /path/to/local/controls/production --model CDCN --model-checks-only --pilot
python scripts/run_architecture_replication.py --project-root . --data-parent /path/to/dataset-parent --output-dir /path/to/local/replication --previous-run-dir /path/to/local/controls/production --model CDCN --limit-targets 1 --seeds 2031 --pilot
python scripts/run_architecture_replication.py --project-root . --data-parent /path/to/dataset-parent --output-dir /path/to/local/replication --previous-run-dir /path/to/local/controls/production --model CDCN
python scripts/run_architecture_replication.py --project-root . --data-parent /path/to/dataset-parent --output-dir /path/to/local/replication --previous-run-dir /path/to/local/controls/production --model DGCNN
python scripts/finish_architecture_replication.py --run-dir /path/to/local/replication --previous-run-dir /path/to/local/controls/production --output-dir /path/to/local/replication/analysis --config configs/architecture_replication.json
```

The extension requires 999 new stage records (819 CDCN and 180 DGCNN source-draw records). Completion analysis rejects missing targets, seeds, or source draws. CDCN uses its fixed Adam configuration and source-derived BatchNorm running statistics during auxiliary forwards; the pilot checks that zero domain loss prevents auxiliary inputs from changing encoder parameters or buffers. Within each target, source draws are averaged within seed and seeds before bootstrap. Backbones use different fixed optimizers and regularization, so their paired differences are procedure-level replication evidence rather than isolated causal architecture effects.

### DREAMER self-rating extension

The separate DREAMER extension uses participant-rated valence and arousal, both backbones, all 23 outer targets, and three training seeds (276 trajectories). It retains every original trial, uses fixed last-60-second stateless DE features, and fixes the high/low threshold at strictly greater than 3 versus 3 or below. All folds contain 17 source-training, five source-validation, and one target person. See `configs/dreamer_checkpoint_extension.json` for the complete rule.

```bash
python scripts/prepare_dreamer_extension.py --data-dir /path/to/licensed/Dreamer --output-dir /path/to/local/dreamer/inputs
python scripts/run_dreamer_extension.py --project-root . --features /path/to/local/dreamer/inputs/dreamer_stateless_features.npz --output-dir /path/to/local/dreamer --config configs/dreamer_checkpoint_extension.json --model DGCNN
python scripts/run_dreamer_extension.py --project-root . --features /path/to/local/dreamer/inputs/dreamer_stateless_features.npz --output-dir /path/to/local/dreamer --config configs/dreamer_checkpoint_extension.json --model CDCN
python scripts/analyze_dreamer_extension.py --run-dir /path/to/local/dreamer --config configs/dreamer_checkpoint_extension.json --output-dir /path/to/local/dreamer/analysis
python scripts/figures/plot_dreamer_extension.py --summary /path/to/local/dreamer/analysis/dreamer_summary.json --output-dir /path/to/local/dreamer/analysis/figures
```

The local preparation audit and trial-to-rating mapping contain licensed data and remain outside this repository. DGCNN's 14-channel instance explicitly uses 64 maps; no 32/62-channel montage is substituted. Source-training and adaptation decompositions remain on SEED/SEED-IV. The two DREAMER tasks share participants/recordings and are not independent cohorts.

### Paired statistical synthesis

After both extensions pass their completion checks, `scripts/analyze_submission_evidence.py` evaluates the fixed 63-contrast inventory with paired mean sign-flip and exact signed-rank diagnostics, global Holm correction, descriptive person-bootstrap intervals, and seed/person sensitivity checks. See [Statistical analysis](docs/STATISTICAL_ANALYSIS.md) for commands, chronology, exact tie/zero handling, and the important dependence assumptions arising from overlapping training populations. All statistical outputs remain local.

The complete evidence plotter is `scripts/figures/plot_submission_evidence.py`. It renders selection/transfer, validation-pool, source/adaptation, instrument-quality, target-distribution, and learning-trajectory plots. The journal numerical companion supplies sanitized derived plot inputs; after extracting it, use:

```bash
python scripts/figures/plot_submission_evidence.py --evidence-dir /path/to/companion/plot_inputs/evidence --architecture-run-dir /path/to/companion/plot_inputs/architecture --dreamer-run-dir /path/to/companion/plot_inputs/dreamer --output-dir /path/to/local/figures
```

The same interface accepts the complete local campaign/synthesis directories. It requires verified coverage and averages seeds inside people. Companion data and rendered plots remain outside this repository.

The analysis refuses incomplete campaigns. It averages training seeds within each outer target before descriptive target-bootstrap resampling; overlapping source-training populations still limit the uncertainty interpretation. CUDA and the locally prepared licensed feature caches are required. The full run saves a substantial checkpoint archive, so choose an output drive with sufficient free space. All generated outputs stay local.

For the DEAP provenance check only:

```bash
python scripts/audit_deap_source_consistency.py --data-dir /path/to/DEAP --output-dir /path/to/local/deap-audit
```

This audit does not edit subject files or certify the origin of a local workbook. Its trial-level outputs can contain licensed ratings and must not be committed. The older reproduction commands below describe preserved historical configurations rather than the revised submission's quantitative scope.

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

For the controlled reruns, render the complete local summary with:

```bash
python scripts/figures/plot_controlled_revision.py --summary /path/to/local/controls/analysis/controlled_summary.json --output-dir /path/to/local/figures
```

The plotter checks complete target/seed coverage before rendering the checkpoint, source-control, adaptation, and target-distribution figures. It writes vector PDFs and PNG previews. Summary files and rendered figures stay outside this public repository.

Architecture and experimental-design schematics can be generated as editable self-contained HTML and SVG with `python scripts/figures/draw_replication_design.py --output-dir /path/to/local/illustrations`. They describe source-code structure and information boundaries, not experimental results.

The older code-only figure utilities are also preserved in `scripts/figures/`. The historical result plotter requires the locally generated `C1_EQUAL_SUBJECT_PRIMARY.csv`, `C1_CANDIDATE_BUDGET_SUMMARY.csv`, `C2_MATCHED_STATS.csv`, `C3_DANN_STATS.csv`, `PER_CLASS_CANONICAL.csv`, and `CANONICAL_RESULT_SUMMARY.csv` in one directory. These derived result files are not distributed here.

```bash
python scripts/figures/plot_result_figures.py --data-dir path/to/local/summaries --output-dir path/to/figures
python scripts/figures/draw_contrast_flow.py --output-prefix path/to/figures/fig_contrast_flow
```

The plotter reads the summary values without retraining models or recomputing metrics; it writes vector PDFs and PNG previews. The flow diagram has no data dependency.

### Cross-person checkpoint transfer

`scripts/analyze_cross_person_checkpoint.py` re-evaluates archived checkpoints on each reserved validation person. One person selects a checkpoint, the remaining validation people select a comparator, and both checkpoints are scored on a different outer target using the saved target epoch curve. SEED and SEED-IV each use 15 outer targets, three reserved validation people, and 40 checkpoints. The FACED DGCNN extension uses 123 outer targets, 12 reserved validation people, and eight checkpoints. Rotations are averaged within each outer target before resampling target means for a descriptive interval. The comparison changes the identity and number of validation people supporting selection; it does not isolate a pure person-identity effect or independent-training uncertainty.

The script requires the complete locally generated 40-epoch SEED and SEED-IV DGCNN or CDCN trajectories and diagnostics. SEED CDCN additionally requires the corrected deterministic-evaluation diagnostics. `--data-parent` is the directory containing `data/SEED/seed_lds_cache.npz` and `data/SEED/seed_annotated_cache.npz`; omit it when those caches are under this repository's `data/` folder. The SEED-IV provider cache remains under `experiments/p4_cross_dataset/` as generated by the preparation script.

FACED requires the eight-epoch strict DGCNN trajectories, the frozen feature cache under `experiments/r2d_faced/cache/`, and locally generated `artifacts/audits/faced_manifest.json` and `faced_trials.csv`. For FACED, `--data-parent` contains `data/FACED/Recording_info.csv`. The cache and audit files must correspond to the archived training runs. The default processes all targets; `--limit-folds` is available for a partial smoke check.

```bash
python scripts/analyze_cross_person_checkpoint.py --dataset SEED --model DGCNN --data-parent /path/to/dataset-parent
python scripts/analyze_cross_person_checkpoint.py --dataset SEED --model CDCN --data-parent /path/to/dataset-parent
python scripts/analyze_cross_person_checkpoint.py --dataset SEED-IV --model DGCNN
python scripts/analyze_cross_person_checkpoint.py --dataset SEED-IV --model CDCN
python scripts/analyze_cross_person_checkpoint.py --dataset FACED --model DGCNN --data-parent /path/to/dataset-parent
python scripts/figures/plot_cross_person.py --input-dir experiments/cross_person_checkpoint --output-prefix experiments/cross_person_checkpoint/figure
```

JSON and CSV outputs go to the ignored `experiments/cross_person_checkpoint/` directory by default. No dataset, metadata, manuscript source, checkpoint, or result file is committed to this repository.

The plotter renders the four 40-checkpoint settings together and a separate FACED panel pair (`figure_faced.pdf`/`.png`) using a scale appropriate to the eight-checkpoint extension.

## License

Original author-written code in this repository is released under the [MIT License](LICENSE). The CDCN evaluation wrapper in `src/tac_revision/models.py` adapts methods from LibEER; its upstream copyright and MIT terms are preserved in [third-party notices](docs/THIRD_PARTY.md). Third-party libraries, benchmark implementations, and datasets remain subject to their respective licenses and terms. This repository does not relicense or redistribute SEED, SEED-IV, FACED, DEAP, LibEER, or other third-party materials. Fold manifests are experiment specifications, not dataset redistribution.
