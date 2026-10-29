# Paired evidence analysis

The synthesis runs only after both the architecture/source-subset and DREAMER campaigns have complete verified outputs. It does not change training, select a favourable seed, or add model fitting. Its chronology is explicit: the earlier DGCNN findings were known before this analysis extension; extension training had also started. This is not a preregistration or a new untouched confirmation cohort.

```bash
python scripts/analyze_submission_evidence.py --architecture-run-dir /path/to/local/replication --dreamer-run-dir /path/to/local/dreamer --config configs/submission_evidence_analysis.json --output-dir /path/to/local/evidence
```

The fixed inventory contains 63 contrasts: ten checkpoint-transfer, 12 source-component, eight adaptation-component, ten transfer-shortfall, eight candidate-set, ten full-pool-versus-single-person, and five implementation-interaction comparisons. FACED's full and grid-eight sets are identical and are not a separate hypothesis. The configuration fixes this inventory before synthesis. Source endpoints and other saved outcomes remain available descriptively.

Source draws and selector rotations are averaged inside seed; three training seeds are averaged inside each target person. The script rejects missing people/seeds and checks its means against the complete campaign summaries. Neither windows nor seeds are additional independent target people. DREAMER's two tasks share all people and recordings and are reported separately.

For each paired person-level contrast the report provides mean/median changes in percentage points, descriptive target-bootstrap 95% intervals, individual-person and seed outcomes, signed rank-biserial effects, threshold fractions, and leave-one-person/seed diagnostics. The mean sign-flip test is exhaustive for at most 20 nonzero differences and otherwise uses 499,999 seeded simulations with a plus-one p-value correction. Simulation-tail uncertainty is recorded with Bonferroni allocation across the 63 planned comparisons. The signed-rank test uses Pratt zeros, average ranks, and a dynamic-programming exact weighted-sum null, including ties. Rank inputs are rounded at 10 decimal places in pp; means/intervals are unrounded. Reference tests check the algorithms against SciPy's untied exact distribution and explicit enumeration with ties/zeros.

Global Holm correction covers all 63 contrasts separately for the two test engines. Within-family adjusted values are supplemental. The analysis does not select whichever engine produces a smaller p-value. It flags agreement, simulation-sensitive decisions and direction reversals; those flags do not automatically establish a claim. The 95% bootstrap intervals are descriptive and are not simultaneous intervals.

**Dependence boundary.** Both person-level tests use a coordinatewise sign-invariance working model for paired differences. Independent symmetric paired differences are sufficient, but shared source training in cross-validation can violate this assumption. Aggregation, exact rank calculations, and Holm correction do not establish independence or provide unconditional coverage for the complete learning procedure. These p-values are conditional diagnostics with explicit assumptions. A full-cohort seed-block sensitivity keeps all target people together within each training seed; only three blocks are available, so its minimum two-sided p-value is 0.25. It is not replaced with a person-seed analysis to obtain a smaller p-value.

No equivalence/noninferiority margin is established; nonsignificance is not evidence of no effect or equivalence. Fractions above 0.5/1/2 pp are descriptive, not validated clinical thresholds. Implementation interactions also include the fixed optimizer/regularization differences. Candidate-set results change both number and composition. Full-pool rules are not selected using target scores.

The local outputs include a complete statistical table, person-seed contrasts, fixed-final instrument-quality summaries, a claim-to-evidence ledger, input hashes, and a review report. They can contain derived results and must remain outside the public repository. Class recalls, confusion summaries, learning histories and source-draw details from the original campaign analyses remain necessary when interpreting weak or heterogeneous instruments.

Reconstruct both window- and trial-level outcomes at fixed-final and full-source-pool-selected checkpoints for every fitted instrument with:

```bash
python scripts/audit_model_learning_quality.py --architecture-run-dir /path/to/local/replication --previous-run-dir /path/to/local/controls/production --dreamer-run-dir /path/to/local/dreamer --config configs/submission_evidence_analysis.json --output-dir /path/to/local/evidence/learning_quality
```

This requires all 1,194 checkpoint fits (459 DGCNN, 459 CDCN, 276 DREAMER). It checks prediction normalization, class support, saved metrics, trial truth, candidate/history alignment and seed/person coverage, and exports class-normalized confusion outcomes, prediction concentration, and every saved training/validation/target curve. History contains the cumulative mean training objective, not instantaneous loss or training accuracy; optimizer/regularization differences prevent interpreting the two implementations' objectives as the same metric. Target curves remain retrospective diagnostics. Weak references and single-class predictions are retained and flagged for interpretation.

Methodological references: [Bengio and Grandvalet (2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html), [Bates, Hastie and Tibshirani](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/), [Holm (1979)](https://www.jstor.org/stable/4615733), and the [SciPy signed-rank](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) and [permutation-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) documentation.
