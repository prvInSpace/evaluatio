from typing import List

from evaluatio.metrics import uer


def test_universal_error_rate():
    """UER should be 1/2 since the reference length is 2, and there is one mistake"""
    res = uer.universal_error_rate([["hello", "world"]], [["hello", "work"]])
    assert res == 0.5


def test_uer_custom_object():
    """UER support custom types. This is used to check that __eq__ is called correctly.
    In this case the object returns true if the value it is being compared with is in
    a list of accepted options"""

    class Option:
        """Allow different options"""

        def __init__(self, values: List[str]) -> None:
            self.values = values

        def __eq__(self, value: object) -> bool:
            return value in self.values

    # Welsh and English version of the same word
    opt = Option(["ffotograff", "photograph"])
    ref = [["I", "have", "a", opt]]
    assert 0.0 == uer.universal_error_rate(ref, [[*ref[0][:-1], "photograph"]])
    assert 0.0 == uer.universal_error_rate(ref, [[*ref[0][:-1], "ffotograff"]])
    assert 0.25 == uer.universal_error_rate(ref, [[*ref[0][:-1], "ice-cream"]])
