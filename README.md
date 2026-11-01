# Cross-subject EEG emotion recognition: evaluation protocols

Code and fixed experiment specifications for controlled evaluation studies in cross-subject EEG emotion recognition on SEED, SEED-IV, FACED and DREAMER. The repository supports reproduction of source-subject allocation, checkpoint selection, target-data access, preprocessing, and subject-level evaluation. Historical DEAP utilities remain for provenance; DEAP quantitative results are excluded from the revised submission because local rating sources could not be independently authenticated. Licensed datasets and previously generated results are not distributed.

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

The controlled experiment runner uses three fixed training seeds and separates checkpoint selection, source-subject composition, available examples, optimizer exposure, and unlabelled target input under a fixed adaptation objective:

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

Architecture, subject-partition/update-budget, and experimental-design schematics can be generated as editable self-contained HTML and SVG with `python scripts/figures/draw_replication_design.py --output-dir /path/to/local/illustrations`. They describe source-code structure and information boundaries, not experimental results.

The older code-only figure utilities are also preserved in `scripts/figures/`. The historical result plotter requires the locally generated `C1_EQUAL_SUBJECT_PRIMARY.csv`, `C1_CANDIDATE_BUDGET_SUMMARY.csv`, `C2_MATCHED_STATS.csv`, `C3_DANN_STATS.csv`, `PER_CLASS_CANONICAL.csv`, and `CANONICAL_RESULT_SUMMARY.csv` in one directory. These derived result files are not distributed here.

```bash
python scripts/figures/plot_result_figures.py --data-dir path/to/local/summaries --output-dir path/to/figures
python scripts/figures/draw_contrast_flow.py --output-prefix path/to/figures/fig_contrast_flow
```

The plotter reads the summary values without retraining models or recomputing metrics; it writes vector PDFs and PNG previews. The flow diagram has no data dependency.

### Cross-subject checkpoint transfer

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

### Validation-split and temporal-model extensions

`configs/split_sensitivity_extension.json` fixes nested, reproducible validation-person sets for each held-out person. `scripts/run_split_sensitivity.py` retrains each model for the same number of updates and checkpoint positions as its corresponding local reference trajectory. Validation-set size changes the people available for training and the resulting per-example exposure; the comparison concerns the whole partitioning policy at fixed compute. Source-validation labels select checkpoints; held-out target labels are used for scoring only. Both DREAMER tasks share the same subject partitions.

The runner requires the original project inputs, LibEER checkout, prepared DREAMER features, and previously generated reference `result.json` files, all stored locally. No recordings, weights or results are included in the repository. For example, using local paths appropriate to the user's installation:

```bash
python scripts/run_split_sensitivity.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir /path/to/new-runs --model DGCNN --dataset SEED --seed 2031
python scripts/analyze_split_sensitivity.py --run-dir /path/to/complete-new-runs --output-dir /path/to/new-analysis
python scripts/analyze_split_quality.py --run-dir /path/to/complete-new-runs --reference-runs-root /path/to/previous-runs --output-dir /path/to/new-analysis
```

`scripts/evaluate_archived_temporal_transfer.py` re-evaluates saved SEED-IV Temporal CNN and FACED Residual TCN checkpoints under person-level balanced-accuracy selection. `scripts/run_temporal_transfer_extension.py` trains three fresh seeds using their source-only partitions, normalizers and fixed configurations. These scripts require the historical raw-window caches, fold specifications and model files; they do not download datasets. `scripts/analyze_temporal_transfer_extension.py` rejects incomplete coverage and summarizes paired person-level outcomes. `scripts/analyze_temporal_quality.py` reconstructs final and selected scores, class concentration and trial-level summaries from local predictions. The temporal models use different representations and optimization settings from the graph and dense-convolutional models.

```bash
python scripts/analyze_temporal_quality.py --run-dir /path/to/temporal-runs --analysis-dir /path/to/temporal-analysis --output-dir /path/to/temporal-quality
python scripts/figures/plot_extension_results.py --kind split --analysis-dir /path/to/new-analysis --output-dir /path/to/figures
python scripts/figures/plot_extension_results.py --kind temporal --analysis-dir /path/to/temporal-analysis --output-dir /path/to/figures
```

The split figure shows the 20 paired differences in selected-minus-final target BAcc for smaller and larger validation partitions relative to the reference partition. It reads the paired means and descriptive 95% subject-bootstrap intervals directly from `split_sensitivity_statistics.json`. Both panels use the same percentage-point scale and label the validation-pool and target-subject counts. The intervals are unadjusted; formal comparisons retain the separate 20-contrast Holm correction. Changing the partition also changes source-training membership and available examples at fixed optimizer updates.

### Selection policies, longer horizons and adversarial schedules

`configs/selection_horizon_extension.json` defines four source-validation rules on the same candidates: peak BAcc, minimum predictive NLL, and patience-five variants of each. Pooled validation gives each subject equal weight. Individual-subject selectors remain secondary. NLL is computed from logits with stable log-softmax; native loss changes and BAcc percentage-point changes are reported separately.

The SEED and SEED-IV extension covers both feature backbones, all 15 held-out subjects and seeds 2031ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2033. It recovers validation loss from the 180 saved trajectories, retrains 180 matched paths to 80 budget units, and requires exact agreement with every saved model state in the first 40 units. Model, optimizer, random-generator and batch-stream states support exact continuation after interruption. The 20/40/80-unit comparison changes training exposure and candidate availability jointly. Patience stopping is emulated from chronological prefixes; its update counts are not measured wall-clock speedups.

The schedule extension adds 720 matched fits. Each ramp is applied to both source-proxy and unlabelled-target arms, keeping domain-loss weight at one. A reversal cap of 0.1 tests schedule shape relative to fixed 0.1; a cap of one also changes magnitude. The scripts do not select a schedule by target performance. All outcomes are retained. The additional 56 paired contrasts have their own global Holm adjustment, alongside raw values and within-question sensitivity. The original 63-comparison adjustment is retained. These are exploratory analyses of previously examined cohorts; overlapping training subjects still limit population inference.

Use a data-parent directory containing `data/SEED/`, not the `SEED` directory itself. The project root supplies the prepared SEED-IV features and the pinned LibEER dependency. Previously generated runs are read only, and all new outputs must be outside this repository.

```bash
python scripts/run_selection_horizon_extension.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir /path/to/extension --model DGCNN --dataset SEED --seed 2031 --phase recovery
python scripts/run_selection_horizon_extension.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir /path/to/extension --model DGCNN --dataset SEED --seed 2031 --phase horizon
python scripts/run_selection_horizon_extension.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir /path/to/extension --model DGCNN --dataset SEED --seed 2031 --phase dann
python scripts/analyze_selection_horizon_extension.py --run-dir /path/to/extension --runs-root /path/to/previous-runs --output-dir /path/to/extension/analysis
```

`supervise_selection_horizon.py` runs the complete queue with at most two workers, an exclusive supervisor lock, first-case reproduction checks and progress logs. A failed deterministic reproduction check stops expansion. An interrupted supervisor adopts its surviving workers rather than launching duplicates. Complete results are required before statistical analysis.

Before supervising, copy the selection-horizon configuration to `OUTPUT/frozen_protocol.json` and use an unchanged code checkout. Reference folders under `--runs-root` are `controlled_revision_20261029/campaign_v1/production` for DGCNN and `architecture_replication_20261030/CDCN/production` for CDCN, followed by `SEED` or `SEEDIV`, `seed_N`, `target_N`, and the `checkpoint` or `adaptation` stage. These are outputs of the corresponding reference runners, not files supplied by this repository.

After complete analysis, run `verify_selection_horizon_extension.py --run-dir OUTPUT --runs-root REFERENCES` to recompute scores from predictions, check stopping chronology and independently enumerate the paired tests and adjustments. It also exports path-free extended learning curves, with interval-mean objectives labelled separately from cumulative means. `scripts/figures/plot_selection_horizon.py --analysis-dir OUTPUT/analysis --output-dir FIGURES` produces vector policy, horizon and schedule plots, plus validation-loss and interval-objective curves when the complete trajectory export is present. Horizon points are offset vertically to keep coincident estimates visible. The renderer rejects incomplete coverage. No significance stars or target-selected schedule are used.

### Binary FACED and DE+linear-SVM comparisons

`configs/binary_faced_svm_extension.json` defines the follow-up. Binary FACED excludes neutral clips, maps the four negative categories to zero and the four positive categories to one, and retains all 123 subjects. DGCNN and CDCN each use three seeds and the same source partitions and eight update positions as their paired nine-class references: 738 trajectories. The update budget remains 968, including for the smaller retained binary training set; it is not described as eight passes over that set.

DE+linear-SVM uses the same eligible feature windows and training/validation/target subjects as the neural references across six task settings. Its feature standardizer fits only the source-training anchors. Twelve ascending C values are evaluated on reserved source subjects, with equal-subject mean balanced accuracy and the first maximum selecting C before target scoring. This yields 322 deterministic outer-subject fits and 3,864 solver fits. Neural results average three seeds within each target; SVM fits are not repeated as additional replicates. Matching concerns data eligibility, information access and window-level scoring. Method-specific scaling, optimization and search budgets differ.

Create the reference campaigns and DREAMER features using the preceding commands, then use an unchanged checkout and copy the binary/SVM configuration to `OUTPUT/frozen_protocol.json`. Reference folders under `--runs-root` follow the layouts described above, with DREAMER outputs in `dreamer_extension_20261030/MODEL/production/TASK`. Outputs and licensed inputs must remain outside this repository.

```bash
python scripts/run_binary_faced.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir OUTPUT --model DGCNN --seed 2031 --gate --targets sub000
python scripts/run_binary_faced.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir OUTPUT --model DGCNN --seed 2031
python scripts/run_matched_svm.py --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --output-dir OUTPUT --dataset FACED-binary --features /path/to/dreamer_stateless_features.npz
```

Repeat the binary runner for both models and all three seeds, and the matched SVM runner for all six configured tasks. The gate compares every saved state and target probability against an existing nine-class trajectory; this checks implementation consistency, not recovery of a published paper's numerical result. `supervise_faced_svm.py` supports separate `neural`, `matched_svm` and `native` phases. Its `--code-root` must point to the unchanged worker checkout containing the copied configuration; it adopts surviving workers after supervisor interruption. GPU concurrency is limited to two, CPU concurrency to four, and exclusive job locks prevent duplicates.

The FACED published-reference branch reconstructs the supplied DE+SVM scripts separately. See the [FACED dataset paper](https://doi.org/10.1038/s41597-023-02650-w) and the [LIBLINEAR reference](https://jmlr.org/papers/v9/fan08a.html). It requires the provider DE files, presentation-order MAT files, processed recordings and supplied reference output CSVs. These materials are not redistributed. The extractor supplies five bands while a normalization script reshapes for four, and the final normalization exclusion differs from the classifier's 15-subject final fold. Both band interpretations and both final-fold prior exclusions are reported. The released whole-record transformations and held-out-label C selection are retained only in this qualified reconstruction, which is not a source-only baseline or certified exact reproduction. Legacy solver convergence warnings remain part of its outputs. Differences from published scores cannot be attributed to a single protocol choice.

```bash
python scripts/prepare_faced_native.py --data-root /path/to/FACED --output-dir OUTPUT
python scripts/run_faced_native.py --input-dir OUTPUT/native/prepared --output-dir OUTPUT --task binary --bands 5
```

Run the native branch for binary/nine-class tasks and five/four bands, giving 44 fold records and 528 C fits. After every expected record is complete:

```bash
python scripts/verify_binary_faced_predictions.py --run-dir OUTPUT --project-root /path/to/project --data-parent /path/to/dataset-parent --runs-root /path/to/previous-runs --model DGCNN
python scripts/verify_faced_svm_predictions.py --run-dir OUTPUT --project-root /path/to/project --data-parent /path/to/dataset-parent --features /path/to/dreamer_stateless_features.npz --kind matched
python scripts/verify_faced_svm_predictions.py --run-dir OUTPUT --project-root /path/to/project --data-parent /path/to/dataset-parent --kind native
python scripts/analyze_binary_faced_svm.py --run-dir OUTPUT --runs-root /path/to/previous-runs --output-dir OUTPUT/analysis
python scripts/verify_faced_svm_statistics.py --run-dir OUTPUT --runs-root /path/to/previous-runs
python scripts/figures/plot_binary_faced_svm.py --analysis-dir OUTPUT/analysis --output-dir /path/to/local/figures
```

Run the neural prediction verifier for each model using the arguments displayed by `--help`. It recomputes every binary source-validation and target curve from saved states. The SVM verifier independently reconstructs source scalers and all validation C curves; its native branch independently reconstructs preprocessing and candidate predictions. Complete follow-up analysis requires all 738 neural, 322 matched SVM and 44 reconstruction records. Independent statistical verification rebuilds all 20 new contrast matrices and preserves the original 143 raw/adjusted results.

The new 20-comparison exploratory family has separate mean and exact Pratt signed-rank Holm adjustments; the original 63/20/4/56 adjustments are unchanged. A joint 163-comparison adjustment is an additional exploratory sensitivity. Overlapping fitted source populations still constrain inference. Window BAcc is the matched comparison endpoint; secondary trial summaries average SVM decision margins and neural probabilities, respectively. All C values, reconstruction choices, outcomes and convergence notices are retained locally. The figure renderer requires complete independently verified results, uses paired-difference intervals and keeps model identities distinguishable by marker shape.

## License

Original author-written code in this repository is released under the [MIT License](LICENSE). The CDCN evaluation wrapper in `src/tac_revision/models.py` adapts methods from LibEER; its upstream copyright and MIT terms are preserved in [third-party notices](docs/THIRD_PARTY.md). Third-party libraries, benchmark implementations, and datasets remain subject to their respective licenses and terms. This repository does not relicense or redistribute SEED, SEED-IV, FACED, DEAP, LibEER, or other third-party materials. Fold manifests are experiment specifications, not dataset redistribution.

## Common-input model comparison

The [comparison workflow](docs/MATCHED_MODEL_COMPARISON.md) provides pinned EEGNet, TSception, RGNN, NSAL-DGAT and MS-MDA interfaces, common-input DGCNN/CDCN/SVM comparators, reference-equivalence checks and resumable subject-disjoint workers. MGGL requires sufficient author-reference information and is not replaced by a proxy network. Keep generated inputs, weights and results in an external working directory.
