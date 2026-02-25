from typing import List

import evaluatio._bindings as _bindings


def word_error_rate_array(predictions: List[str], references: List[str]) -> List[float]:
    """Calculates the word level error-rate for every zipped pair of predictions and references.
    The delimiter used to split the words is ' '.

    NOTE: If the reference string is empty or contain no words, the resulting WER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.word_error_rate_array(predictions, references)


def word_edit_distance_array(
    predictions: List[str], references: List[str]
) -> List[int]:
    """Calculates the word level edit-distance for every pair in predictions and references.
    The delimiter used to split the words is ' '.

    NOTE: If the reference string is empty or contain no words, the resulting WER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.word_edit_distance_array(predictions, references)


def word_error_rate(predictions: List[str], references: List[str]) -> float:
    """Calculates the mean word level error-rate for the entire set.
    This is the equivalent of using the `wer` metric for the `evaluate` library (using `jiwer`).
    The delimiter used to split the words is ' '.

    NOTE: If the reference string is empty or contain no words, the resulting WER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.word_error_rate(predictions, references)
