# Third-party model reference

The graph and dense convolutional model implementations are imported from [LibEER](https://github.com/XJTU-EEG/LibEER), revision `dddff9776dbdae21195fe320dff0a5ba61628a18`. LibEER is MIT-licensed, copyright 2024 xjtu-eeg. Its source is not vendored here. Its [license at the pinned revision](https://github.com/XJTU-EEG/LibEER/blob/dddff9776dbdae21195fe320dff0a5ba61628a18/LICENSE) governs its code independently.

From the repository root:

```bash
git clone https://github.com/XJTU-EEG/LibEER.git tmp/p3_references/LibEER
git -C tmp/p3_references/LibEER checkout dddff9776dbdae21195fe320dff0a5ba61628a18
```

The local code imports the pinned source from `tmp/p3_references/LibEER/LibEER`. Keep its license and citation when using that dependency. The `CorrectedConvblock.forward` and `CorrectedCDCN.forward` methods in `src/tac_revision/models.py` adapt the corresponding methods of `LibEER/models/CDCN.py` to make dropout respect evaluation mode. The file carries an upstream notice; the full upstream license is retained in `docs/THIRD_PARTY_LICENSES/LibEER_LICENSE.txt`. No LibEER source file is overwritten.

`src/p4_replication/deap_features.py` mirrors the pinned LibEER DEAP preprocessing sequence and imports its feature-extraction functions; `src/p3_protocol_benchmark/protocol.py` mirrors its documented split rule. These are protocol-matching local implementations rather than redistributed copies of LibEER files. Dataset recordings and provider features are not included.

`scripts/run_controlled_revision.py` subclasses the pinned DGCNN to fix its parameter settings explicitly, uses its graph-convolution/laplacian operations, and attaches the pinned discriminator and gradient-reversal operation. The controlled domain wrapper preserves the DGCNN B1ReLU encoder/classifier in every adaptation arm. It is a specified control implementation, distinct from the released `DannDgcnn` forward path, which uses different feature activations. The upstream MIT license above applies to the reused components.

`scripts/run_architecture_replication.py` reuses the corrected CDCN module and the same pinned discriminator/reversal components. Its CDCN wrapper exposes the existing representation before final classification dropout and freezes encoder BatchNorm running statistics during auxiliary passes. The schematics in `scripts/figures/draw_replication_design.py` are original code-based illustrations of these pinned implementations; they do not reproduce artwork from the source papers.
