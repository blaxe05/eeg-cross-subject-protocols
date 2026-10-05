from src.strict_eeg_benchmark.splits import assert_strict_loso_fold, make_loso_folds


def test_loso_folds_are_subject_disjoint_and_complete():
    subjects = tuple(map(str, range(1, 16)))
    folds = make_loso_folds(subjects, seed=7, validation_fraction=0.2)
    assert {fold.held_out_subject for fold in folds} == set(subjects)
    for fold in folds:
        assert_strict_loso_fold(fold, set(subjects))
        assert fold.held_out_subject not in fold.source_train_subjects
        assert fold.held_out_subject not in fold.source_validation_subjects
