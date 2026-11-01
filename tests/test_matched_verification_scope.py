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
