"""
Universal error metrics

This module provides utilities to compute universal error rate (UER) and
universal edit distance between reference and hypothesis sequences.
Unlike ``metrics.wer`` and ``metrics.cer``, all computations operate on arbitrary iterables of
tokens rather than whitespace-tokenized strings, making this module suitable
for subword-level or any other custom tokenization scheme.

The functions accept any iterable of iterables and internally convert them
to a format compatible with the underlying native bindings.

Notes
-----
- If a reference sequence is empty or contains no tokens, the corresponding
  UER is defined as ``inf``.
- These functions are thin wrappers around optimized native implementations.
"""

from typing import Any, Iterable, List

from evaluatio import _bindings
from evaluatio.inference.ci import ConfidenceInterval, _convert_confidence_interval


def universal_error_rate_per_pair(
    references: Iterable[Iterable[Any]], hypotheses: Iterable[Iterable[Any]]
) -> List[float]:
    """
    Compute universal error rate (UER) for each reference-hypothesis pair.

    Parameters
    ----------
    references : Iterable[Iterable[Any]]
        Iterable of reference token sequences.
    hypotheses : Iterable[Iterable[Any]]
        Iterable of hypothesis token sequences. Must be the same length as
        ``references``.

    Returns
    -------
    List[float]
        Universal error rate for each pair of reference and hypothesis.

    Raises
    ------
    ValueError
        If the lists are of different lengths.

    See Also
    --------
    metrics.cer.word_error_rate_per_pair : Character-tokenized string version.
    metrics.wer.word_error_rate_per_pair : Whitespace-tokenized string version.

    Notes
    -----
    - Tokens are compared using ``__eq__`` if types differ.
    - If a reference sequence is empty or contains no tokens, the resulting
      UER is ``inf``.
    """
    return _bindings.universal_error_rate_per_pair(references, hypotheses)


def universal_edit_distance_per_pair(
    references: Iterable[Iterable[Any]], hypotheses: Iterable[Iterable[Any]]
) -> List[int]:
    """
    Compute universal edit distance for each reference-hypothesis pair.

    Parameters
    ----------
    references : Iterable[Iterable[Any]]
        Iterable of reference token sequences.
    hypotheses : Iterable[Iterable[Any]]
        Iterable of hypothesis token sequences. Must be the same length as
        ``references``.

    Returns
    -------
    List[int]
        Edit distance for each pair of reference and hypothesis.

    See Also
    --------
    metrics.cer.word_edit_distance_per_pair : Character-tokenized string version.
    metrics.wer.word_edit_distance_per_pair : Whitespace-tokenized string version.

    Notes
    -----
    - Tokens are compared using ``__eq__`` if types differ.
    """
    return _bindings.universal_edit_distance_per_pair(references, hypotheses)


def universal_error_rate(
    references: Iterable[Iterable[Any]], hypotheses: Iterable[Iterable[Any]]
) -> float:
    """
    Compute the corpus level universal error rate (UER) over all pairs.

    Parameters
    ----------
    references : Iterable[Iterable[Any]]
        Iterable of reference token sequences.
    hypotheses : Iterable[Iterable[Any]]
        Iterable of hypothesis token sequences. Must be the same length as
        ``references``.

    Returns
    -------
    float
        Corpus level universal error rate across all pairs.

    See Also
    --------
    metrics.cer.word_error_rate : Character-tokenized string version.
    metrics.wer.word_error_rate : Whitespace-tokenized string version.

    Notes
    -----
    - Tokens are compared using ``__eq__`` if types differ.
    - If all reference sequences are empty or contain no tokens, the resulting
      UER is ``inf``.
    """
    return _bindings.universal_error_rate(references, hypotheses)


def universal_error_rate_ci(
    references: Iterable[Iterable[Any]],
    hypotheses: Iterable[Iterable[Any]],
    iterations: int,
    alpha: float,
) -> ConfidenceInterval:
    """
    Estimate a confidence interval for the universal error rate via bootstrapping.

    Parameters
    ----------
    references : Iterable[Iterable[Any]]
        Iterable of reference token sequences.
    hypotheses : Iterable[Iterable[Any]]
        Iterable of hypothesis token sequences. Must be the same length as
        ``references``.
    iterations : int
        Number of bootstrap iterations.
    alpha : float
        Significance level for the confidence interval.

    Returns
    -------
    ConfidenceInterval
        Estimated confidence interval for the corpus level universal error rate.

    See Also
    --------
    metrics.cer.word_error_rate_ci : Character-tokenized string version.
    metrics.wer.word_error_rate_ci : Whitespace-tokenized string version.

    Notes
    -----
    - The bootstrapped metric corresponds to ``universal_error_rate``.
    - Tokens are compared using ``__eq__`` if types differ.
    - If any reference sequence is empty or contains no tokens, the resulting
      UER can be ``inf``.
    """
    return _convert_confidence_interval(
        _bindings.universal_error_rate_ci(references, hypotheses, iterations, alpha)
    )
