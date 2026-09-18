import numpy as np
import pandas as pd

from charge_key_automation.matching import derive_description, hybrid_match


def test_exact_match_wins():
    ref = pd.DataFrame({"chrg_code": ["A", "B"], "chrg_desc": ["THEFT", "ASSAULT"]})
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = hybrid_match("A", "THEFT", ref, ["THEFT", "ASSAULT"], emb, emb[0], 0.85, 90)
    assert result.method == "exact"
    assert result.score == 1.0


def test_attempt_description_derivation():
    mapping = pd.Series({"187": "HOMICIDE"})
    desc, derived = derive_description("664/187", mapping)
    assert derived is True
    assert desc == "ATT:HOMICIDE"
