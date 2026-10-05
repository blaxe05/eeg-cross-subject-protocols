import numpy as np
import pytest

from src.strict_eeg_benchmark.leakage import LeakageError
from src.strict_eeg_benchmark.preprocessing import AuditedPCA, AuditedSelectKBest, AuditedStandardScaler
from src.strict_eeg_benchmark.training import select_model
from src.strict_eeg_benchmark.types import FeatureBatch, FoldSubjects, Partition


FOLD = FoldSubjects("3", ("1",), ("2",), seed=11)


def batch(subject: str, partition: Partition) -> FeatureBatch:
    return FeatureBatch(
        X=np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]]),
        y=np.asarray([0, 1, 0, 1]),
        subject_ids=np.asarray([subject] * 4),
        session_ids=np.asarray(["s1"] * 4),
        trial_ids=np.asarray(["t1", "t2", "t3", "t4"]),
        dataset_name="synthetic",
        feature_names=("a", "b"),
        partition=partition,
    )


@pytest.mark.parametrize(
    "transformer",
    [AuditedStandardScaler(), AuditedPCA(n_components=1), AuditedSelectKBest(k=1)],
)
def test_target_data_cannot_fit_any_transformation(transformer):
    with pytest.raises(LeakageError):
        transformer.fit(batch("3", Partition.TARGET_TEST), FOLD)


def test_validation_data_cannot_fit_scaler():
    with pytest.raises(LeakageError):
        AuditedStandardScaler().fit(batch("2", Partition.SOURCE_VALIDATION), FOLD)


def test_target_data_cannot_drive_model_selection():
    train = batch("1", Partition.SOURCE_TRAIN)
    target = batch("3", Partition.TARGET_TEST)
    with pytest.raises(LeakageError):
        select_model("logistic_regression", train, target, FOLD, [0.1, 1.0], seed=11)
    with pytest.raises(LeakageError):
        select_model("logistic_regression", target.with_partition(Partition.SOURCE_TRAIN), batch("2", Partition.SOURCE_VALIDATION), FOLD, [0.1, 1.0], seed=11)
