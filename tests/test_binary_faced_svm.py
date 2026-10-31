import sys
from pathlib import Path
import numpy as np
from sklearn.svm import LinearSVC

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from faced_svm_common import binary_panel,config
from run_faced_native import lds
from run_matched_svm import decision_scores
from src.tac_revision.provider_training import Panel
from scipy.stats import rankdata
from verify_faced_svm_statistics import exact_pratt,direct_mean_test,direct_holm
from analyze_submission_evidence import paired_summary
from analyze_binary_faced_svm import classwise_rows,native_candidate_summary


def test_binary_mapping_retains_subjects_trials_and_feature_values():
    labels=np.r_[np.repeat(np.arange(4),3),np.repeat(4,4),np.repeat(np.arange(5,9),3)]
    y=np.tile(np.repeat(labels,30),123)
    subject=np.repeat([f'sub{i:03d}' for i in range(123)],840)
    trial=np.repeat(np.arange(123*28).astype(str),30)
    local=np.tile(np.arange(30),123*28)
    x=np.broadcast_to(np.arange(len(y),dtype=np.float32)[:,None,None],(len(y),32,5))
    before=Panel(x,y,subject,trial,local,32,9,'synthetic')
    after=binary_panel(before)
    assert np.array_equal(after.x[:,0,0],np.flatnonzero(y!=4))
    assert np.array_equal(after.y,(y[y!=4]>=5).astype(int))
    assert len(set(after.subject))==123 and len(set(after.trial))==2952


def test_svm_scores_preserve_classifier_predictions_for_two_and_many_classes():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(90,6))
    for classes in [2,3,9]:
        y=np.arange(90)%classes
        clf=LinearSVC(dual=False,tol=1e-8,max_iter=20000).fit(x,y)
        scores=decision_scores(clf,x)
        assert scores.shape==(90,classes)
        assert np.array_equal(scores.argmax(axis=1),clf.predict(x))


def test_lds_initial_mean_uses_the_complete_sequence():
    result=lds(np.array([[0.],[2.]]))
    assert np.isclose(result[0,0],1/1.01)
    assert np.array_equal(lds(np.full((30,5),3.)),np.full((30,5),3.))


def test_frozen_counts_and_new_multiplicity_family_are_distinct():
    value=config()
    assert len(value['svm_c_grid'])==12
    assert value['expected_neural_paths']==2*3*123
    assert value['expected_matched_svm_outer_fits']==sum(value['subject_counts'].values())==322
    family=value['new_hypothesis_family']
    assert sum(v for k,v in family.items() if k not in ['name','size'])==family['size']==20
    assert value['preserve_original_hypothesis_families']==[63,20,4,56]


def test_pratt_integer_generating_function_matches_enumerated_signs_with_ties_and_zeros():
    from itertools import product
    for x in [np.array([0.,1.,-1.,2.,0.,2.,-3.]),np.zeros(5),np.array([1.,1.,1.]),np.array([-2.,0.,-2.,1.])]:
        ranks=rankdata(abs(x),method='average');keep=x!=0
        if keep.any():
            observed=abs(np.dot(np.sign(x[keep]),ranks[keep]))
            null=[abs(np.dot(signs,ranks[keep])) for signs in product([-1,1],repeat=int(keep.sum()))]
            expected=np.mean(np.asarray(null)>=observed)
        else:expected=1.
        assert np.isclose(exact_pratt(x),expected,atol=1e-15,rtol=0)


def test_independent_mean_test_matches_exact_and_chunked_monte_carlo_engines():
    cfg={'training_seeds':[2031,2032,2033],'bootstrap_seed':20261029,'bootstrap_replicates':100,
         'signflip_seed':20261030,'exact_signflip_max_nonzero_pairs':20,'signflip_monte_carlo_samples':8193,
         'alpha':.05,'expected_hypotheses':20,'rank_difference_rounding_decimals_pp':10,
         'descriptive_thresholds_pp':[.5,1,2],'p_value_scope':'conditional'}
    for x in [np.zeros(8),np.array([0.,1.,-1.,2.]),np.linspace(-1,3,25)]:
        result=paired_summary(np.repeat(x[:,None],3,axis=1),cfg,'unit-test')
        p,interval=direct_mean_test(x,cfg,'unit-test')
        assert p==result['mean_signflip']['p']
        np.testing.assert_allclose(interval,result['mean_signflip']['mc_tail_interval'],atol=1e-15,rtol=0)
    np.testing.assert_allclose(direct_holm([.01,.015,.04]),[.03,.03,.04],atol=1e-15,rtol=0)


def test_classwise_export_retains_imbalance_and_separates_recall_from_accuracy():
    y=np.array([0,0,0,1]);predicted=np.array([0,1,1,1])
    rows=classwise_rows('example','classifier','subject',7,y,np.eye(2)[predicted])
    assert [r['true_windows'] for r in rows]==[3,1]
    assert [r['predicted_windows'] for r in rows]==[1,3]
    np.testing.assert_allclose([r['recall_pct'] for r in rows],[100/3,100],rtol=0,atol=1e-12)
    np.testing.assert_allclose([r['f1_pct'] for r in rows],[50,50],rtol=0,atol=1e-12)


def test_native_selection_preserves_earliest_exact_global_accuracy_tie():
    counts=np.array([[1,1,3],[0,0,5]])
    predictions=np.ones((2,3,7),dtype=int)
    for candidate in range(2):
        for subject in range(3):predictions[candidate,subject,:counts[candidate,subject]]=0
    scores,chosen=native_candidate_summary(predictions,np.zeros(7,dtype=int))
    np.testing.assert_allclose(scores,counts/7,atol=0,rtol=0)
    assert chosen==0
