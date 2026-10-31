"""The fixed-reference endpoint effect is one contrast per setting."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_selection_horizon_extension import added_contrast_vectors


def test_contrast_inventory_and_schedule_interaction():
    design={'models':['DGCNN'],'datasets':['SEED'],'targets':['1'],'training_seeds':[2031],
            'policies':['bacc_max','nll_min','bacc_patience','nll_patience'],
            'adversarial_schedules':{'ramp_0_1':.1,'ramp_1_0':1.}}
    key=('DGCNN','SEED','1',2031)
    outcomes=[]
    for h,offset in [(20,-1),(40,0),(80,2)]:
        for i,p in enumerate(design['policies']):
            outcomes.append({'horizon':h,'aggregation':'equal_subject_pool','policy':p,
                             'target_change_pp':10*i+offset})
    curve=[{'balanced_accuracy':.4} for _ in range(80)];curve[-1]={'balanced_accuracy':.6}
    records={key:{'outcomes':outcomes,'target_curve':curve}}
    fixed={key:{'arms':[{'arm':'loss_on_target','window':{'balanced_accuracy':.5}},
                        {'arm':'loss_on_proxy','window':{'balanced_accuracy':.45}}]}}
    adapted={key+(schedule,arm):{'window':{'balanced_accuracy':value}}
             for schedule in design['adversarial_schedules']
             for arm,value in [('loss_on_target',.65),('loss_on_proxy',.55)]}
    vectors=added_contrast_vectors(records,records,adapted,fixed,design)
    assert len(vectors)==14
    terminal=[r for r in vectors if r[0]=='terminal_80_minus_40']
    assert len(terminal)==1 and terminal[0][-1][0,0]==pytest.approx(20)
    assert [r[-1][0,0] for r in vectors if r[0]=='selector_at_40']==[10,20,30]
    for r in vectors:
        if r[0]=='schedule_interaction':
            np.testing.assert_allclose(r[-1],[[5.]])
