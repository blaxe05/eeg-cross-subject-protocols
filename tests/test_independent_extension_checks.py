from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_selection_horizon_extension import exact_tests,direct_holm


def test_enumeration_including_zero_pairs():
    assert exact_tests([1.,2.,3.]) == (.25,.25)
    assert exact_tests([-1.,0.,1.]) == (1.,1.)
    assert exact_tests([0.,0.,0.]) == (1.,1.)


def test_adjustment_restores_original_order():
    np.testing.assert_allclose(direct_holm([.03,.01,.02,.5]),[.06,.04,.06,.5])
