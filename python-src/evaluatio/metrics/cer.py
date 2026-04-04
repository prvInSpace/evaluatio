"""
Character-level error metrics

This module provides utilities to compute character error rate (CER) and
character-level edit distance between reference and hypothesis text sequences.

The functions accept any iterable of strings and internally convert them
to a format compatible with the underlying native bindings.

Notes
-----
- If a reference string is empty, the corresponding CER is defined as ``inf``.
- These functions are thin wrappers around optimized native implementations.
"""

from typing import Iterable, List

from evaluatio import _bindings
from evaluatio.inference.ci import ConfidenceInterval, _convert_confidence_interval


def character_error_rate_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> List[float]:
    """
    Compute character error rate (CER) for each reference-hypothesis pair.

    Parameters
    ----------
    references : Iterable[str]
        Iterable of reference strings.
    hypotheses : Iterable[str]
        Iterable of hypothesis strings. Must be the same length as
        ``references``.

    Returns
    -------
    List[float]
        Character error rate for each pair of reference and hypothesis.
    
    Raises
    ------
    ValueError
        If the lists are of different lengths.

    See Also
    --------
    metrics.uer.universal_error_rate_per_pair : Type-agnostic version.

    Notes
    -----
    - If a reference string is empty or contains no characters, the resulting
      CER is ``inf``.
    """
    return _bindings.character_error_rate_per_pair(references, hypotheses)


def character_edit_distance_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> List[int]:
    """
    Compute character-level edit distance for each reference-hypothesis pair.

    Parameters
    ----------
    references : Iterable[str]
        Iterable of reference strings.
    hypotheses : Iterable[str]
        Iterable of hypothesis strings. Must be the same length as
        ``references``.

    Returns
    -------
    List[int]
        character-level edit distance for each pair.
    """
    return _bindings.character_edit_distance_per_pair(references, hypotheses)


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """
    Compute the corpus level character error rate (CER) over all pairs.

    Parameters
    ----------
    references : Iterable[str]
        Iterable of reference strings.
    hypotheses : Iterable[str]
        Iterable of hypothesis strings. Must be the same length as
        ``references``.

    Returns
    -------
    float
        Corpus level character error rate across all pairs.

    Notes
    -----
    - Equivalent to common CER implementations (e.g., ``jiwer``-based metrics).
    - If all reference strings are empty, the resulting CER is ``inf``.
    """
    return _bindings.character_error_rate(references, hypotheses)


def character_error_rate_ci(
    references: Iterable[str], hypotheses: Iterable[str], iterations: int, alpha: float
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
    iterations : int
        Number of bootstrap iterations.
    alpha : float
        Significance level for the confidence interval.

    Returns
    -------
    ConfidenceInterval
        Estimated confidence interval for the corpus level character error rate.

    Notes
    -----
    - The bootstrapped metric corresponds to ``character_error_rate``.
    - If any reference string is empty or contains no characters, the resulting
      CER can be ``inf``.
    """
    return _convert_confidence_interval(
        _bindings.character_error_rate_ci(references, hypotheses, iterations, alpha)
    )
