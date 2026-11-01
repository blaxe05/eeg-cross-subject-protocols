"""Evidence certification requires complete verification, even after training ends."""
from pathlib import Path
import sys
import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
from verify_matched_model_results import verification_report


def passed(*identities):return [{'id':key,'status':'pass'} for key in identities]


def test_sample_cannot_certify_a_completed_training_campaign():
    report=verification_report('sample',['a','b'],passed('a','b'),[])
    assert report['supported_outputs_complete']
    assert report['records_verified']==report['records_planned']==2
    assert report['complete_campaign'] is False


@pytest.mark.parametrize('checks,missing',[(passed('a'),[]),(passed('a'),['b'])])
def test_complete_scope_requires_every_identity(checks,missing):
    with pytest.raises(ValueError,match='requires every planned identity'):
        verification_report('complete',['a','b'],checks,missing)


@pytest.mark.parametrize('checks,missing',[
    (passed('a','a'),[]),
    (passed('a','outside'),[]),
    (passed('a'),['a']),
    ([{'id':'a','status':'failed'}],[]),
])
def test_invalid_records_cannot_enter_a_passing_certificate(checks,missing):
    with pytest.raises(ValueError):verification_report('sample',['a','b'],checks,missing)


def test_full_verification_certificate_is_identity_complete():
    report=verification_report('complete',['a','b'],passed('b','a'),[])
    assert report['complete_campaign'] is True
    assert report['scope']=='supported_jobs_only'
    assert report['records_verified']==2 and report['supported_records_still_missing']==0


def test_component_certification_keeps_global_campaign_incomplete():
    report=verification_report('component',['a','b','c'],passed('a','b'),['c'],['a','b'])
    assert report['complete_component'] is True and report['complete_campaign'] is False
    assert report['records_planned']==3 and report['component_records_verified']==2


def test_component_mode_never_certifies_entire_campaign():
    report=verification_report('component',['a','b'],passed('a','b'),[],['a','b'])
    assert report['supported_outputs_complete'] and report['complete_campaign'] is False


@pytest.mark.parametrize('component,checks,missing',[
    (['a','b'],passed('a'),['b']),
    (['a','b'],passed('a'),[]),
    (['a'],passed('a','b'),[]),
    (['outside'],passed('a'),[]),
    (['a','a'],passed('a'),[]),
    ([],passed('a'),[]),
])
def test_component_scope_rejects_missing_or_misassigned_evidence(component,checks,missing):
    with pytest.raises(ValueError):verification_report('component',['a','b'],checks,missing,component)
