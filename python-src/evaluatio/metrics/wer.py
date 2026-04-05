"""
Word-level error metrics

This module provides utilities to compute word error rate (WER) and
word-level edit distance between reference and hypothesis text sequences.
All computations operate on whitespace-tokenized words. If you need more
complex tokenizing, please see ``metrics.uer``.

The functions accept any iterable of strings and internally convert them
to a format compatible with the underlying native bindings.

Notes
-----
- If a reference string is empty or contains no tokens, the corresponding
  WER is defined as ``inf``.
- These functions are thin wrappers around optimized native implementations.
"""

from typing import Iterable, List

from evaluatio import _bindings
from evaluatio.inference.ci import ConfidenceInterval, _convert_confidence_interval


def word_error_rate_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> List[float]:
    """
    Compute word error rate (WER) for each reference-hypothesis pair.

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
        Word error rate for each pair of reference and hypothesis.

    Raises
    ------
    ValueError
        If the lists are of different lengths.

    See Also
    --------
    metrics.uer.universal_error_rate_per_pair : Type-agnostic version.

    Notes
    -----
    - Tokenization is performed by splitting on whitespace.
    - If a reference string is empty or contains no tokens, the resulting
      WER is ``inf``.
    """
    return _bindings.word_error_rate_per_pair(references, hypotheses)


def word_edit_distance_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> List[int]:
    """
    Compute word-level edit distance for each reference-hypothesis pair.

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
        Word-level edit distance for each pair.

    See Also
    --------
    metrics.uer.universal_edit_distance_per_pair : Type-agnostic version.

    Notes
    -----
    - Tokenization is performed by splitting on whitespace.
    """
    return _bindings.word_edit_distance_per_pair(references, hypotheses)


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """
    Compute the corpus level word error rate (WER) over all pairs.

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
        Corpus level word error rate across all pairs.

    Notes
    -----
    - Tokenization is performed by splitting on whitespace.
    - Equivalent to common WER implementations (e.g., ``jiwer``-based metrics).
    - If all reference strings are empty or contains no tokens, the resulting
      WER is ``inf``.
    """
    return _bindings.word_error_rate(references, hypotheses)


def word_error_rate_ci(
    references: Iterable[str], hypotheses: Iterable[str], iterations: int, alpha: float
) -> ConfidenceInterval:
    """
    Estimate a confidence interval for the word error rate via bootstrapping.

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
        Estimated confidence interval for the corpus level word error rate.

    Notes
    -----
    - The bootstrapped metric corresponds to ``word_error_rate``.
    - Tokenization is performed by splitting on whitespace.
    - If any reference string is empty or contains no tokens, the resulting
      WER can be ``inf``.
    """
    return _convert_confidence_interval(
        _bindings.word_error_rate_ci(references, hypotheses, iterations, alpha)
    )
