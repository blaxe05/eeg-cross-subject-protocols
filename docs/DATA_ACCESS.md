# Data access and preparation

Obtain SEED, SEED-IV, and FACED directly from their providers under the applicable terms. DEAP is retained only for the local provenance audit and historical reproduction; its model results are excluded from the revised submission. No raw recordings, provider feature arrays, licensed ratings, results, or manuscript files are included here.

## Controlled-study preparation

The preparation utility reads the licensed files and refuses to overwrite an existing prepared output:

```bash
python scripts/prepare_controlled_inputs.py --dataset SEED --data-parent /path/to/dataset-parent
python scripts/prepare_controlled_inputs.py --dataset SEED-IV --data-parent /path/to/dataset-parent --project-root .
python scripts/prepare_controlled_inputs.py --dataset FACED --data-parent /path/to/dataset-parent --project-root .
```

`--data-parent` contains `data/SEED`, `data/SEED-IV`, and `data/FACED`. The SEED utility writes its two derived caches in the source SEED directory by default; use `--seed-cache-dir` to choose a separate cache location. The controlled runner's SEED `--data-parent` must contain those prepared caches. SEED-IV and portable FACED features are written below the selected project's ignored `experiments/` directory. FACED's portable cache includes its subject/trial index, so it does not require private audit files from earlier runs.

SEED preparation reads `Preprocessed_EEG/label.mat` and `ExtractedFeatures_1s/*_YYYYMMDD.mat`, explicitly selecting the provider's `de_LDS` arrays. SEED-IV uses its licensed smoothed-feature files and provider session-specific labels. FACED requires `Recording_info.csv`, `Stimuli_info.xlsx`, `Task_event.xlsx`, and `Processed_data/sub000.pkl` through `sub122.pkl`; it rebuilds stateless Welch DE with the archived float32 input boundary. The optional `--faced-smoke-subjects` output is labelled partial and is rejected as a model input.

The preparation path has been checked against all archived SEED and SEED-IV model-facing arrays and metadata, and all 123 FACED subjects (103,320 windows), including feature arrays, labels, subjects, trials, and local window indices. All compared arrays agree exactly. This establishes preparation equivalence for the local files; it is distinct from independent authentication of a provider's files. Keep the generated arrays and audit metadata local.

## DREAMER extension inputs

Obtain authorized access to DREAMER from its provider ([record](https://zenodo.org/records/546113); [original paper](https://doi.org/10.1109/JBHI.2017.2688239)). The extension reads the local `DREAMER.mat` version 1.0.2 and, when present, its accompanying documentation. Embedded electrode order, sampling rate, trial counts, score ranges, finite signals, and per-person binary class support are checked before preparation. Features use the last 60 seconds of each stimulus in non-overlapping one-second windows; baseline recordings are not used. No local population-dependent preprocessing is fitted.

`prepare_dreamer_extension.py` writes a local feature cache, aggregate preparation audit, and private trial-to-rating map. Do not commit them. A local checksum identifies the copy used; the currently restricted provider record does not expose anonymous file checksums, so internal checks alone are not independent authentication. Valence/arousal labels come directly from the same MAT participant/trial structure, with high strictly greater than 3 and low including 3. No trial/subject is excluded by the declared preparation.

The HBUED files are not inputs to this extension. Their local sampling-rate and channel-order provenance has not been established for these controls.

## Preserved historical inputs

The one-fold runners expect data beneath a local, ignored `data/` directory. The recorded runs used the following inputs:

| Dataset | Local input used by the code | Preparation / provenance |
|---|---|---|
| SEED | `data/SEED/seed_lds_cache.npz`, `seed_annotated_cache.npz` | Provider DE-LDS arrays and aligned subject/session/trial metadata; `scripts/audit_p3_libeer.py` checks provider feature identity. |
| SEED-IV | `data/SEED-IV/` and `experiments/p4_cross_dataset/seediv_provider_de_lds.npz` | `scripts/prepare_p4_seediv.py` builds the provider-feature cache from licensed files. |
| FACED | `data/FACED/` plus the documented window index and DE cache | `src/strict_eeg_benchmark/r2_data.py` and `r2d_core.py` define trial-safe indexing and the fixed feature representation. |
| DEAP | `data/DEAP/data/s01.dat` … `s32.dat`, licensed `participant_ratings.xls` | `scripts/prepare_p4_deap.py` builds trial-local features; `scripts/audit_deap_revision_labels.py` verifies the workbook-to-trial mapping and produces local, ignored labels. |

Historical DEAP reruns used workbook-derived labels after inconsistencies were found in local subject-file valence/arousal entries. They are not quantitative evidence for the revised submission. The workbook and per-trial label tables must remain local. A matching checksum establishes local file identity, not independent provider authentication. The new `scripts/audit_deap_source_consistency.py` describes the exact local transformation pattern without rewriting either source or declaring one independently authenticated.

The complete training runs also require the fixed files and intermediate caches named by the relevant runner. An absent cache produces an explicit error. Do not substitute a different representation or redraw subject folds when reproducing the manuscript results.
