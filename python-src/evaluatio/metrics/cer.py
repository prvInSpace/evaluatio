"""Character error rate (CER) functions"""

from typing import Iterable

import evaluatio._bindings as _bindings


def character_error_rate_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> Iterable[float]:
    """Calculates the character level error-rate for every zipped pair of references and hypotheses.

    NOTE: If the reference string is empty or contain no characters, the resulting CER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.character_error_rate_per_pair(references, hypotheses)


def character_edit_distance_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> Iterable[int]:
    """Calculates the character level edit-distance for every zipped pair of references and hypotheses.

    NOTE: If the reference string is empty or contain no characters, the resulting CER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.character_edit_distance_per_pair(references, hypotheses)


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Calculates the mean word level error-rate for the entire set.
    This is the equivalent of using the `cer` metric for the `evaluate` library (using `jiwer`)

    NOTE: If the reference string is empty or contain no characters, the resulting CER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.character_error_rate(references, hypotheses)
