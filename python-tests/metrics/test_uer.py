from typing import List

from evaluatio.metrics import uer

class Option:
    """Allow different options"""

    def __init__(self, values: List[str]) -> None:
        self.values = values

    def __eq__(self, value: object) -> bool:
        return value in self.values

def test_universal_error_rate():
    """UER should be 1/2 since the reference length is 2, and there is one mistake"""
    res = uer.universal_error_rate([["hello", "world"]], [["hello", "work"]])
    assert res == 0.5


def test_uer_custom_object():
    """UER support custom types. This is used to check that __eq__ is called correctly.
    In this case the object returns true if the value it is being compared with is in
    a list of accepted options"""
    # Welsh and English version of the same word
    opt = Option(["ffotograff", "photograph"])
    ref = [["I", "have", "a", opt]]
    assert 0.0 == uer.universal_error_rate(ref, [[*ref[0][:-1], "photograph"]])
    assert 0.0 == uer.universal_error_rate(ref, [[*ref[0][:-1], "ffotograff"]])
    assert 0.25 == uer.universal_error_rate(ref, [[*ref[0][:-1], "ice-cream"]])

def test_uer_per_pair_custom_object():
    """UER support custom types. This is used to check that __eq__ is called correctly.
    In this case the object returns true if the value it is being compared with is in
    a list of accepted options"""
    # Welsh and English version of the same word
    opt = Option(["ffotograff", "photograph"])
    ref = [["I", "have", "a", opt]]
    assert 0.0 == uer.universal_error_rate_per_pair(ref, [[*ref[0][:-1], "photograph"]])[0]
    assert 0.0 == uer.universal_error_rate_per_pair(ref, [[*ref[0][:-1], "ffotograff"]])[0]
    assert 0.25 == uer.universal_error_rate_per_pair(ref, [[*ref[0][:-1], "ice-cream"]])[0]


def test_ued_custom_object():
    """UER support custom types. This is used to check that __eq__ is called correctly.
    In this case the object returns true if the value it is being compared with is in
    a list of accepted options"""
    # Welsh and English version of the same word
    opt = Option(["ffotograff", "photograph"])
    ref = [["I", "have", "a", opt]]
    assert 0 == uer.universal_edit_distance_per_pair(ref, [[*ref[0][:-1], "photograph"]])[0]
    assert 0 == uer.universal_edit_distance_per_pair(ref, [[*ref[0][:-1], "ffotograff"]])[0]
    assert 1 == uer.universal_edit_distance_per_pair(ref, [[*ref[0][:-1], "ice-cream"]])[0]

def test_uer_ci():
    references = [
        ["hello", "world"],
        ["cymru", "am", "byth"],
        ["croeso", "i", "gymru"]
    ]
    predictions = [
        ["hello", "byd"], # 50%
        ["gymru", "ap", "dydd"], # 100%
        ["croeso", "i", "gymru"] # 0%
    ]
    res = uer.universal_error_rate_ci(references, predictions, 1000, 0.01)
    
    # Since we have so many iterations it is almost impossible for this not
    # to be the case, but I guess it could fail if all resamples left out one
    # of the three cases but that is incredibly unlikely.
    assert res.lower < res.mean < res.upper
    assert res.mean == 0.5
    assert res.lower <= 0.125
    assert res.upper >= 0.875