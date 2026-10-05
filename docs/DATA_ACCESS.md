# Data access and preparation

Obtain SEED, SEED-IV, FACED, and DEAP directly from their providers under the applicable terms. No raw recordings, provider feature arrays, licensed ratings, results, or manuscript files are included here.

The one-fold runners expect data beneath a local, ignored `data/` directory. The recorded runs used the following inputs:

| Dataset | Local input used by the code | Preparation / provenance |
|---|---|---|
| SEED | `data/SEED/seed_lds_cache.npz`, `seed_annotated_cache.npz` | Provider DE-LDS arrays and aligned subject/session/trial metadata; `scripts/audit_p3_libeer.py` checks provider feature identity. |
| SEED-IV | `data/SEED-IV/` and `experiments/p4_cross_dataset/seediv_provider_de_lds.npz` | `scripts/prepare_p4_seediv.py` builds the provider-feature cache from licensed files. |
| FACED | `data/FACED/` plus the documented window index and DE cache | `src/strict_eeg_benchmark/r2_data.py` and `r2d_core.py` define trial-safe indexing and the fixed feature representation. |
| DEAP | `data/DEAP/data/s01.dat` … `s32.dat`, licensed `participant_ratings.xls` | `scripts/prepare_p4_deap.py` builds trial-local features; `scripts/audit_deap_revision_labels.py` verifies the workbook-to-trial mapping and produces local, ignored labels. |

The DEAP workbook-derived labels, not the inconsistent local subject-file valence/arousal entries, define the two reported tasks. The workbook and per-trial label tables must remain local. Record checksums and aggregate counts in your local experiment record; a matching checksum establishes local file identity, not independent provider authentication.

The complete training runs also require the fixed files and intermediate caches named by the relevant runner. An absent cache produces an explicit error. Do not substitute a different representation or redraw subject folds when reproducing the manuscript results.
