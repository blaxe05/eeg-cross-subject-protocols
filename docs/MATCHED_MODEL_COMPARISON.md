# Common-input model comparison

The comparison uses subject-disjoint training, source validation and target scoring. Raw-input models receive one-second signal windows; feature models receive stateless differential entropy derived from those same windows. Normalization is fitted on training anchors only. Equal-subject source-validation balanced accuracy selects checkpoints, with earliest ties. Target labels enter only after checkpoint choices have been saved.

The supported models are EEGNet, TSception, RGNN, NSAL-DGAT and MS-MDA, together with common-input DGCNN/CDCN and linear-SVM comparators. MGGL is reserved pending sufficient author-reference information. The software deliberately raises an error for MGGL instead of substituting a different network. Published scores are separate from executions of these programs.

The six task settings are SEED, SEED-IV, FACED nine-class/binary and DREAMER valence/arousal. SEED uses the first session; SEED-IV uses all sessions. The common FACED inputs retain the first 30 scalp-labelled channels and exclude the two HEOR/HEOL columns in the supplied 32-column array. Earlier analyses using that full array remain separate. DREAMER retains the last 60 stimulus seconds, with high labels defined by ratings greater than three.

The earlier supplied SEED-IV features have four-second support, as specified by the [dataset provider](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html). Their counts are therefore not interchangeable with one-second raw-window counts. The common-input comparison constructs new features from the aligned raw windows; it does not equate the two representations.

## Reference software

Fetch the exact revisions listed in `scripts/fetch_matched_references.py` to a directory outside this repository. Upstream code remains subject to its own terms and is not vendored here. Reference model code, generated indices, data, weights and results belong in the external working directory.

- EEGNet: [author architecture](https://github.com/vlawhern/arl-eegmodels); the PyTorch port preserves grouped depthwise filters, temporal kernel duration, layer order, flatten order and max-norm constraints. An independent NumPy forward calculation checks the port.
- TSception: [author network](https://github.com/yi-ding-cs/TSception); bilateral pairs are ordered left then right, excluding midline/unpaired channels as required by its spatial filters.
- RGNN: [author graph implementation](https://github.com/zhongpeixiang/RGNN); a dense, signed-degree equivalent is tested against its outputs and gradients. The graph-only arm is source-only; optional node-wise RevGrad arms use either unlabelled target signals or a matched source proxy. Binary supervision follows the reference one-hot rule. Nine-class label smoothing is an explicit extension because the reference provides no nine-class emotion-distance matrix.
- NSAL-DGAT: [author model](https://github.com/YYingDL/NSAL-DGAT); fixes explicitly preserve electrode/band axes, parameterize dataset dimensions, save otherwise unregistered graph/bank state, preserve training mode during bank updates, handle zero-degree graph columns, supply logits to a logits-based loss and replace global adaptive max pooling with an equivalent deterministic operation. The encoder/classifier is checked against the original reference. These are qualified reference-based runs, not a claim of exact published-score reproduction.
- MS-MDA: [author implementation](https://github.com/VoiceBeer/MS-MDA); retains the shared extractor, every source-specific extractor/classifier, marginal-distribution and cross-branch discrepancy terms, and probability averaging. Its loss, input gradients and ensemble outputs are checked against the reference. Input and class dimensions are parameterized for the declared datasets.

NSAL-DGAT and MS-MDA target arms are unlabelled-target adaptation procedures. Source-proxy arms retain their objectives and auxiliary-example counts. They are not silently described as source-only versions of the published methods. Multi-source batches remain domain-specific, while other methods use pooled source batches; architecture-level causal attribution is inappropriate. Domain-specific batches wrap within the source subject when there are fewer unique anchors than the batch size; unique anchors and processed-example counts are reported separately.

## Execution

1. Install the repository requirements and acquire each dataset through its provider. Create the usual dataset indices and raw-window caches with the dataset preparation tools. Keep these local.
2. Fetch the pinned reference software with `scripts/fetch_matched_references.py`.
3. Run `scripts/prepare_matched_models.py` to align raw/DE windows and verify raw-cache alignment. It requires the local project inputs and the pinned LibEER directory for the anatomical SEED prior.
4. Run `scripts/make_matched_model_jobs.py` to freeze folds, update budgets, seeds and roles. `configs/matched_model_budgets.json` supplies optimizer-update hyperparameters, not scores. Existing trajectory metadata can optionally verify those budgets; reproducing the design does not require another researcher's result files. The generated protocol/jobs files belong in the local working directory.
5. Run `scripts/verify_matched_model_implementations.py`, the focused unit tests, and `scripts/pilot_matched_models.py`. Pilots train on real source inputs but never score target labels. The process-restart check requires identical candidate-state hashes.
6. Freeze source/configuration hashes, then run `scripts/supervise_matched_models.py`. Workers use exclusive group leases, complete optimizer/RNG/sampler/memory-bank checkpoints and atomic records. All candidate states and prediction arrays are retained. A supervisor restart adopts matching live workers.
7. Run `scripts/verify_matched_model_results.py` with the frozen `--source-dir`, the external `--references` directory and an external `--report` path. `--mode sample` checks available records without certifying a campaign. `--mode complete` requires every supported job and re-evaluates every neural candidate on the full validation and target inputs. Both modes independently reconstruct training-only normalizers, source choices, window/trial metrics and SVM candidate decisions. Use complete verification before statistical synthesis.

Verification reports distinguish completed training outputs from verified evidence. Sample mode never certifies a complete campaign, even when all output files exist. Complete certification requires every planned supported job identity; reports also retain hashes of the protocol, job specification, frozen-source manifest and checked result records.

Once all SVM outputs exist, `--mode component --component SVM` verifies every classical-model result on the CPU while neural jobs continue. Its certificate covers the SVM component and keeps `complete_campaign=false`. Add `--refit-svm-examples` to repeat the full C grid for the first outer fold in each task and compare fitted coefficients/intercepts. These are verification fits, separate from the planned experimental records. The checks also require the declared C grid, saved source-only selections, correct fold/row identities and zero recorded convergence warnings. Complete campaign verification remains required before the combined analysis.

Use the supplied CLI `--help` for each program's arguments. The local supervisor expects passing implementation, unit-test and pilot records and a matching source manifest before releasing production jobs. Windows runs use sequential MKL and a single OpenMP runtime; the unsupported duplicate-runtime override is not used.

The optional `--scheduling-config PATH` changes operational concurrency and queue priority without changing frozen workers, folds, seeds, inputs, optimizer updates or candidate positions. For example, the JSON below permits two nine-class FACED workers and one binary FACED worker, with feature models first in the binary queue:

```json
{
  "gpu_workers": 3,
  "dataset_gpu_limits": {"FACED": 2, "FACED-binary": 1},
  "priority_datasets": ["FACED-binary"],
  "dataset_model_priority": {"FACED-binary": ["DGCNN", "RGNN", "NSAL-DGAT", "CDCN", "EEGNet", "TSception", "MS-MDA"]}
}
```

Dataset-specific model priorities apply across seeds; unlisted models retain their normal relative order. The file is reloaded each poll. Write changes atomically. Invalid or temporarily unreadable changes pause new launches and retain every live worker. Lowered limits take effect as existing groups finish; workers are never killed to satisfy a new limit. Dataset limits are caps, so revise them when the other dataset finishes to avoid leaving available capacity idle. Without a configuration file, the original queue order and CLI concurrency apply.

Additional workers share the same GPU. Free memory alone does not establish a speedup: compare observed combined throughput, GPU memory and worker durations before increasing limits, and preserve the sequential-runtime and concurrency records separately from scientific results. A supervisor may be restarted after confirming its PID and command line; the replacement must adopt existing workers using the original immutable `--source-dir` and existing leases.

Balanced accuracy, accuracy, Macro-F1 and trial summaries are calculated from held-out predictions. Seed and validation-selector repetitions are averaged within target subjects. Follow-up tests use separate prospective families and leave earlier adjustments intact; overlapping source-training populations still constrain inference. Identical update counts do not establish convergence or equal wall-clock compute.

## Analysis input checks and comparison definitions

`scripts/matched_analysis_common.py` supplies tested building blocks for the final synthesis. Its input check requires a complete independent verification report covering every supported job identity and matching the current protocol, job, source-manifest and result hashes. A complete SVM component or sample report cannot satisfy this requirement. The helpers do not constitute a completed statistical analysis.

Individual and pooled checkpoint choices use source-validation BAcc only, with earliest ties and equal subject weights. Individual-selector effects are averaged within each target/seed. Seed effects are then averaged within each target subject before inference; neither selectors nor seeds are additional subjects. Selection-minus-transfer is the mean source-selection gain minus the corresponding selected-minus-final target change, in percentage points.

The planned comparison inventory retains 144 checkpoint, 60 procedure-performance and 18 auxiliary-population slots. Procedure-performance differences compare pooled source-BAcc-selected window BAcc between the declared model/reference arms. Auxiliary-population differences compare final checkpoints in target and source-proxy arms. SVM scores provide descriptive context, without an added retrospective hypothesis family. The original 163 comparisons and their separate corrections remain unchanged.

MGGL-dependent slots remain missing unless a defensible implementation becomes available. Without MGGL, there are 192 supported new contrasts and 30 unresolved slots. Holm calculations retain the planned family sizes; arithmetic padding is never exported as an observed effect or p-value. Raw and adjusted probabilities must accompany all observed contrasts. Descriptive subject-bootstrap intervals and conditional paired tests do not establish independence of overlapping source-training populations, equivalence from nonsignificance, or optimal architecture performance.

After complete prediction verification, `scripts/analyze_common_input_models.py` reconstructs selector effects, checks paired conditions and exports every supported target/seed vector, absolute outcome and raw/adjusted test. `configs/matched_model_analysis.json` fixes aggregation, descriptive bootstrap and conditional test settings. Absolute SVM summaries contain one deterministic fit per subject; neural summaries average three seeds within that subject.

```bash
python scripts/analyze_common_input_models.py --run-dir /path/to/complete-campaign --prediction-proof /path/to/complete-prediction-proof.json --source-dir /path/to/frozen-training-source --output-dir /path/to/new-analysis
```

The program rejects incomplete manifests, component-only verification and changed result/array hashes. Its report explicitly marks statistical calculations as awaiting independent verification and keeps the six-requested-model scope incomplete while MGGL is unresolved. Independently reconstruct the exported statistics and review interpretation before using findings in a manuscript. It does not modify earlier results or their correction families.

`scripts/common_input_statistical_checks.py` provides independent numerical checks without importing the synthesis helpers or its statistical engine. Exact mean tests enumerate signed sums; exact Pratt tests count subset sums using an integer generating polynomial. Monte Carlo draws retain the declared deterministic seeds and family-specific tail bounds. Separate calculations check descriptive intervals, subject/seed summaries, raw tests and planned-size Holm corrections. Tests include literal sign enumeration, ties, zero differences and deliberately altered summaries. These numerical checks alone do not verify reconstruction of experimental contrast vectors or certify the complete analysis.

`scripts/verify_common_input_statistics.py` combines those numerical checks with an independent reconstruction from saved predictions. It calculates class confusion counts, source choices, target changes and window/trial outcomes without importing production metric or synthesis helpers. It checks all supported comparison vectors, absolute exports, raw/adjusted tests, missing-model slots, labels, counts and file hashes. A complete prediction certificate and complete supported synthesis are required first.

```bash
python scripts/verify_common_input_statistics.py --run-dir /path/to/complete-campaign --analysis-dir /path/to/new-analysis --prediction-proof /path/to/complete-prediction-proof.json --source-dir /path/to/frozen-training-source --synthesis-source-dir /path/to/frozen-analysis-source --report /path/to/statistical-verification.json
```

The verifier has an exclusive report lock and publishes its certificate only after all checks pass. Keep its source and report alongside the local analysis. Passing this check covers the supported numerical evidence; interpretation review remains necessary, and unresolved MGGL requirements still prevent completion of the six-model request.

## Result figures

`scripts/figures/plot_common_input_models.py` requires that complete statistical certificate and matching hashes for all five analysis artifacts. It exports vector PDF/SVG figures and PNG previews with consistent 9-point typography. Six aligned task panels retain common scales and fixed model order. Open/filled markers and distinct shapes supplement color, and horizontal intervals are the descriptive 95% subject-bootstrap intervals.

```bash
python scripts/figures/plot_common_input_models.py --analysis-dir /path/to/new-analysis --verification-report /path/to/statistical-verification.json --output-dir /path/to/local/figures
```

The figures display all 84 individual/pooled checkpoint-transfer estimates, 48 procedure-performance contrasts and 18 auxiliary-population contrasts. The additional 42 shortfall contrasts remain in the complete statistical tables. Source-only and TA-U roles are stated explicitly; all outcomes and both signs are retained. No significance stars or published-score rankings are used. Existing Figures 6 and 7 are not modified by this renderer. Generated inputs, figures and numerical outputs remain outside the code repository.
