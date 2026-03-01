from math import isnan

import evaluatio.metrics.wer as wer

def test_word_error_rate():
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
