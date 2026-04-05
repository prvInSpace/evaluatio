"""
Point-of-interest (POI) evaluation metrics for sequence comparison.

This module implements metrics inspired by the PIER (Point of Interest Error Rate)
algorithm introduced in:

    "PIER: A Novel Metric for Evaluating What Matters in Code-Switching"
    by Ugan et al.

The metrics focus evaluation on specific positions (points of interest) within
a sequence, allowing selective comparison between reference and hypothesis
sequences.

Note that the algorithm explained by Ugan et al. works on a word token level, while
this version takes an iterable of any type. Hence, if you want the same results
you have to pre-tokenize your strings.

All functions are thin wrappers around optimized native bindings.
"""

from typing import Iterable, List

import evaluatio._bindings as _bindings


def poi_edit_distance(
    references: Iterable[object],
    hypotheses: Iterable[object],
    points_of_interest: Iterable[bool],
) -> List[int]:
    """
    Compute edit distance restricted to points of interest.

    Parameters
    ----------
    references : Iterable[object]
        Reference sequence.
    hypotheses : Iterable[object]
        Hypothesis sequence.
    points_of_interest : Iterable[bool]
        Boolean mask indicating which indices should be included in the
        computation. Must be the same length as ``references``.

    Returns
    -------
    List[int]
        Edit distance computed over positions where
        ``points_of_interest`` is ``True``.

    Notes
    -----
    - Only positions marked as ``True`` in ``points_of_interest`` are
      considered when computing the edit distance.
    - The behavior for mismatched lengths follows the underlying
      implementation in the native bindings.
    - This function is a generalised version of the algorithm described by
      Ugan et al.[1]_

    References
    ----------
    .. [1] Ugan, E. Y., Pham, N. Q., Bärmann, L., & Waibel, A. (2025, April).
        Pier: A novel metric for evaluating what matters in code-switching.
        In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) (pp. 1-5). IEEE.
    """
    return _bindings.poi_edit_distance(references, hypotheses, points_of_interest)


def poi_error_rate(
    references: Iterable[Iterable[object]],
    hypotheses: Iterable[Iterable[object]],
    points_of_interest: Iterable[Iterable[bool]],
) -> float:
    """
    Compute point-of-interest (POI) error rate over multiple sequences.

    Parameters
    ----------
    references : Iterable of Iterable[object]
        Collection of reference sequences.
    hypotheses : Iterable of Iterable[object]
        Collection of hypothesis sequences. Must align with ``references``.
    points_of_interest : Iterable of Iterable[bool]
        Collection of boolean masks indicating points of interest for each
        sequence. Each mask must match the length of its corresponding
        reference sequence.

    Returns
    -------
    float
        Corpus level error rate computed over all sequences, restricted to
        points of interest.

    Notes
    -----
    - Only positions marked as ``True`` contribute to the error rate.
    - This function is a generalised version of the algorithm described by
      Ugan et al.[1]_

    References
    ----------
    .. [1] Ugan, E. Y., Pham, N. Q., Bärmann, L., & Waibel, A. (2025, April).
        Pier: A novel metric for evaluating what matters in code-switching.
        In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) (pp. 1-5). IEEE.
    """
    return _bindings.poi_error_rate(references, hypotheses, points_of_interest)
