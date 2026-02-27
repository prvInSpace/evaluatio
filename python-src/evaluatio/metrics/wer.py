from typing import List

import evaluatio._bindings as _bindings


def word_error_rate_per_pair(
    references: List[str], hypotheses: List[str]
) -> List[float]:
    """Calculates the word level error-rate for every zipped pair of references and hypotheses.
    The delimiter used to split the words is ' '.

    NOTE: If the reference string is empty or contain no words, the resulting WER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.word_error_rate_per_pair(references, hypotheses)


def word_edit_distance_per_pair(
    references: List[str], hypotheses: List[str]
) -> List[int]:
    """Calculates the word level edit-distance for every pair in references and hypotheses.
    The delimiter used to split the words is ' '.

    NOTE: If the reference string is empty or contain no words, the resulting WER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.word_edit_distance_per_pair(references, hypotheses)


def word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """Calculates the mean word level error-rate for the entire set.
    This is the equivalent of using the `wer` metric for the `evaluate` library (using `jiwer`).
    The delimiter used to split the words is ' '.

    NOTE: If the reference string is empty or contain no words, the resulting WER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.word_error_rate(references, hypotheses)
