from math import isnan

import evaluatio.metrics.cer as cer
import pytest


def test_character_error_rate():
    res = cer.character_error_rate(["aaaaaa"], ["aaabbb"])
    assert res == 0.5


def test_character_error_rate_no_reference_length():
    """character error rate without any reference lengths should be nan"""
    res = cer.character_error_rate([""], [""])
    assert isnan(res)


def test_character_error_rate_empty():
    """character error rate without any data is also nan"""
    res = cer.character_error_rate([], [])
    assert isnan(res)


def test_character_error_rate_different_lengths():
    """character error rate without any data is also nan"""
    with pytest.raises(ValueError):
        _ = cer.character_error_rate(["hello world"], [])


def test_character_error_rate_per_pair():
    res = cer.character_error_rate_per_pair(["aaaaaa", "aaa"], ["aaabbb", "aaa"])
    assert res == [0.5, 0.0]


def test_character_error_rate_per_pair_no_reference_length():
    res = cer.character_error_rate_per_pair([""], [""])
    assert len(res) == 1
    assert isnan(res[0])


def test_character_error_rate_per_pair_empty():
    res = cer.character_error_rate_per_pair([], [])
    assert len(res) == 0


def test_character_error_rate_per_pair_different_lengths():
    with pytest.raises(ValueError):
        _ = cer.character_error_rate_per_pair(["hello"], [])


def test_character_edit_distance():
    res = cer.character_edit_distance_per_pair(["aa"], ["ab"])
    assert res == [1]


def test_character_edit_distance_no_reference_length():
    res = cer.character_edit_distance_per_pair([""], [""])
    assert res == [0]


def test_character_edit_distance_empty():
    res = cer.character_edit_distance_per_pair([], [])
    assert res == []


def test_character_edit_distance_different_lengths():
    with pytest.raises(ValueError):
        _ = cer.character_edit_distance_per_pair(["hello world"], [])
