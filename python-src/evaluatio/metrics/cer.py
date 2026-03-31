"""Character error rate (CER) functions"""

from typing import Iterable, List

from evaluatio import _bindings
from evaluatio.inference.ci import ConfidenceInterval, _convert_confidence_interval


def character_error_rate_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> List[float]:
    """Calculates the character level error-rate for every zipped pair of references and hypotheses.

    NOTE: If the reference string is empty or contain no characters, the resulting CER is inf

    NOTE: Even though the type indicates that the function only takes lists, it takes any iterable
    that can be converted to a Vec<&string> by pyo3.
    """
    return _bindings.character_error_rate_per_pair(references, hypotheses)


def character_edit_distance_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> List[int]:
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


def character_error_rate_ci(
    references: Iterable[str], hypotheses: Iterable[str], interations: int, alpha: float
) -> ConfidenceInterval:
    """
    Estimate a confidence interval for the character error rate via bootstrapping.

    Parameters
    ----------
    references : Iterable[str]
        Iterable of reference strings.
    hypotheses : Iterable[str]
        Iterable of hypothesis strings. Must be the same length as
        ``references``.
    interations : int
        Number of bootstrap iterations.
    alpha : float
        Significance level for the confidence interval.

    Returns
    -------
    ConfidenceInterval
        Estimated confidence interval for the mean character error rate.

    Notes
    -----
    - The bootstrapped metric corresponds to ``character_error_rate``.
    - If any reference string is empty or contains no characters, the resulting
      CER is ``inf``.
    """
    return _convert_confidence_interval(
        _bindings.character_error_rate_ci(references, hypotheses, interations, alpha)
    )
