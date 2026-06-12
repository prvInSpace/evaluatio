from typing import Iterable, List

from evaluatio import _bindings

def optimal_alignment(
    references: Iterable[object],
    hypotheses: Iterable[object],
) -> List[_bindings.Alignment]:
    """
    Compute an optimal alignment between a reference and hypothesis sequence.

    Aligns each element in ``references`` to a span in ``hypotheses`` using
    dynamic programming, minimising the total edit distance between the two
    sequences. The result describes, for each reference index, which range of
    hypothesis indices it aligns to.

    Parameters
    ----------
    references : Iterable[object]
        Reference token sequence.
    hypotheses : Iterable[object]
        Hypothesis token sequence.

    Returns
    -------
    List[Alignment]
        One ``Alignment`` per reference token, sorted by reference index.
        Each ``Alignment`` has three fields:

        - ``index`` : int — index of the token in ``references``.
        - ``start`` : int — start index of the aligned span in ``hypotheses`` (inclusive).
        - ``end`` : int — end index of the aligned span in ``hypotheses`` (exclusive).

    Notes
    -----
    - Tokens are compared using the same underlying mechanism as universal_edit_distance.
    - Multiple optimal alignments might exist, one is returned arbitrarily, but deterministically.
    """
    return _bindings.optimal_alignment(references, hypotheses)