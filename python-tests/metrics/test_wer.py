from math import isnan

import evaluatio.metrics.wer as wer
import pytest


def test_word_error_rate():
    """Word error rate should be 1/2 since the reference length is 2, and there is one mistake"""
    res = wer.word_error_rate(["hello world"], ["hello work"])
    assert res == 0.5


def test_word_error_rate_no_reference_length():
    """Word error rate without any reference lengths should be nan"""
    res = wer.word_error_rate([""], [""])
    assert isnan(res)


def test_word_error_rate_empty():
    """Word error rate without any data is also nan"""
    res = wer.word_error_rate([], [])
    assert isnan(res)


def test_word_error_rate_different_lengths():
    """Word error rate without any data is also nan"""
    with pytest.raises(ValueError):
        _ = wer.word_error_rate(["hello world"], [])


def test_word_error_rate_per_pair():
    res = wer.word_error_rate_per_pair(
        ["hello world", "croeso"], ["hello work", "croeso"]
    )
    assert res == [0.5, 0.0]


def test_word_error_rate_per_pair_no_reference_length():
    """Word error rate without any reference lengths should be nan"""
    res = wer.word_error_rate_per_pair([""], [""])
    assert len(res) == 1
    assert isnan(res[0])


def test_word_error_rate_per_pair_empty():
    """Word error rate per pair without any pairs should be an empty list"""
    res = wer.word_error_rate_per_pair([], [])
    assert res == []


def test_word_error_rate_per_pair_different_lengths():
    """Word error rate without any data is also nan"""
    with pytest.raises(ValueError):
        _ = wer.word_error_rate_per_pair(["hello world"], [])


def test_word_edit_distance():
    """Word edit distance should be 1 since there is one mistake"""
    res = wer.word_edit_distance_per_pair(["hello world"], ["hello work"])
    assert res == [1]


def test_word_edit_distance_no_reference_length():
    """Word edit distance should be 0"""
    res = wer.word_edit_distance_per_pair([""], [""])
    assert res == [0]


def test_word_edit_distance_empty():
    """Word edit distance without any data should be empty"""
    res = wer.word_edit_distance_per_pair([], [])
    assert res == []


def test_word_edit_distance_different_lengths():
    """Word error rate without any data is also nan"""
    with pytest.raises(ValueError):
        _ = wer.word_edit_distance_per_pair(["hello world"], [])

def test_word_error_rate_ci():
    ref = ["a", "a a", "a"]
    hyp = ["a", "a b", "b"]
    res = wer.word_error_rate_ci(ref, hyp, 1000, 0.01)
    assert res.lower < res.mean < res.upper
    assert res.mean == 0.5
    assert res.lower <= 0.125
    assert res.upper >= 0.875