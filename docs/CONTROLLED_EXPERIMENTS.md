# Controlled experiment specification

The fixed specification is `configs/controlled_protocol_revision.json`. It defines three training seeds (2031–2033), one DGCNN backbone, and the complete target scope. Results are generated locally and are not distributed in this repository.

## Checkpoint comparison

Every SEED and SEED-IV outer target (15 each) and every FACED outer target (123) is evaluated. Strict training/reserved/target partitions are 11/3/1 and 110/12/1, respectively. SEED uses session one; SEED-IV uses all three sessions. Checkpoints are saved at 40 equally spaced update counts for SEED/SEED-IV and eight for FACED, including the prescribed final update.

One reserved person selects the earliest checkpoint maximizing their balanced accuracy. That checkpoint is compared with the same final checkpoint on the selection person and on the different outer target. Selector rotations are averaged within each target. A separate selector uses mean per-person BAcc across the reserved validation people. Target-oracle maxima are retrospective diagnostics and do not select deployable models.

## Source-training path

SEED/SEED-IV use four arms with the same initialization, optimizer, representation, target, and batch size:

| Arm | People | Available examples | Updates |
|---|---|---|---|
| A | Original 11 | Strict pool N | S |
| B | All 14 non-target | Exact same per-class counts as A | S |
| C | Same 14 | Expanded pool containing B | S |
| D | Same 14 | Identical expanded pool | More than S |

C and D share their complete optimizer prefix. The three paired components are B−A, C−B, and D−C. They describe this expansion path rather than universal additive effects. A/B change the identities and diversity of source people, not subject count in isolation. Expanded class proportions match the strict pool up to integer rounding. Fixed-size batches ensure fixed updates also match cumulative processed examples.

## Adaptation controls

All arms retain the DGCNN B1ReLU encoder/classifier and the same binary domain head. With source batches, auxiliary pool sizes, updates, and initialization paired, compare domain loss off with a source proxy, domain loss on with that proxy, and domain loss on with real unlabelled target EEG. The active loss coefficient is 1.0 and gradient reversal coefficient is 0.1. Target emotion labels are absent from the fitting function. Final-step scoring avoids an additional checkpoint-selection contrast.

The pilot also replaces proxy data with real target EEG at zero domain loss and checks identical encoder/classifier parameters. This sampled implementation check is separate from the full production comparisons.

## Aggregation and checks

The full campaign contains 639 stage records: 459 checkpoint, 90 source, and 90 adaptation records. The analysis rejects missing targets or seeds. It averages seeds within each target before 20,000 target-bootstrap draws (seed 20261029). The intervals are descriptive and do not resample overlapping source-training populations. Three seeds are not additional people or a full estimate of training uncertainty.

`verify_controlled_revision.py` checks saved partitions, initializers, example/class/update matching, nested data pools, and available scoring arrays. `analyze_controlled_revision.py` generates the local summary; `scripts/figures/plot_controlled_revision.py` renders vector figures from it.

DEAP performance is outside this quantitative scope. The read-only audit utility describes local rating inconsistencies without certifying source authenticity or rewriting inputs. HSLT is not part of this specification.

## Exploratory validation-pool diagnostic

`scripts/analyze_validation_pool_size.py` reuses the 459 complete checkpoint records after the principal outcomes have been inspected. It averages all validation-person subsets at sizes 1/2/3 for SEED/SEED-IV and 1/2/3/6/12 for FACED, using the same candidate trajectories and final baseline. Subsets and seeds are averaged within target before target resampling. Single-person and full-pool endpoints are checked against the original selectors. This is an explicitly exploratory sensitivity analysis, not a target-tuned deployment rule or an additional training campaign.

```bash
python scripts/analyze_validation_pool_size.py --run-dir /path/to/local/controls/production --output-dir /path/to/local/pool-analysis
python scripts/figures/plot_validation_pool_size.py --summary /path/to/local/pool-analysis/validation_pool_summary.json --output-dir /path/to/local/figures
```

## Architecture and subset replication

`configs/architecture_replication.json` fixes a CDCN extension with the same outer folds, anchors, update counts, candidates, and three training seeds. CDCN uses growth rate 12, dense blocks [6,6,6], dropout 0.5, and Adam (learning rate 0.001, weight decay 0.005, epsilon 0.0001). Evaluation dropout is disabled. Hyperparameters are not selected using target outcomes. Its optimizer and regularization differ from DGCNN, limiting causal interpretation of cross-backbone score differences.

The source controls use three nested-pool draws per training seed and target. Draw 0 reuses the original seed+71 pool rule. Draws 1 and 2 add 100003 times the draw index only to the subset-sampling seed. The initialization and minibatch-position streams remain fixed within seed. Draws and seeds are averaged inside targets; they are not treated as additional people.

CDCN adaptation retains one encoder, classifier, discriminator, reversal coefficient, and domain-loss coefficient across arms. The source forward updates encoder BatchNorm running statistics; auxiliary forwards use those source-derived statistics without updating them. Dropout remains active in training. Pilot checks require identical encoder parameters and buffers, as well as identical final probabilities, when source-proxy inputs are replaced by real target features at zero domain loss. This declared boundary is distinct from a model that adapts BatchNorm statistics to target inputs.

The extension comprises 459 CDCN checkpoint records, 270 CDCN source-draw records, 90 CDCN adaptation records, and 180 DGCNN source-draw records. `finish_architecture_replication.py` requires all 999 new records, verifies stored predictions and controls, averages source draws before target-level uncertainty, and renders local plots. It also repeats candidate-set, validation-pool, class-recall, confusion, and prediction-unit summaries. Primary and exploratory chronology remains explicit; this is an extension after inspection of the earlier DGCNN campaign, not an untouched confirmation study.
