# Third-party model reference

The graph and dense convolutional model implementations are imported from [LibEER](https://github.com/XJTU-EEG/LibEER), revision `dddff9776dbdae21195fe320dff0a5ba61628a18`. LibEER is MIT-licensed by its authors and is not vendored here.

From the repository root:

```bash
git clone https://github.com/XJTU-EEG/LibEER.git tmp/p3_references/LibEER
git -C tmp/p3_references/LibEER checkout dddff9776dbdae21195fe320dff0a5ba61628a18
```

The local code imports the pinned source from `tmp/p3_references/LibEER/LibEER`. Keep its license and citation when using that dependency. `src/tac_revision/models.py` contains the documented CDCN evaluation correction and compatibility wrapper used for the manuscript; no LibEER source file is overwritten.
