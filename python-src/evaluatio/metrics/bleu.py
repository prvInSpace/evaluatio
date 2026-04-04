"""
Bootstrap significance testing for BLEU score comparison.

This module provides a paired bootstrap significance test for comparing two
machine translation systems using the BLEU metric. It follows the method
introduced by Koehn (2004), in which corpus-level BLEU is recomputed on
each bootstrap resample rather than aggregating per-sentence scores directly.

Sufficient statistics (clipped n-gram counts, total n-gram counts, and
lengths required for the brevity penalty) are precomputed per sentence using
sacrebleu, ensuring that tokenisation and scoring are fully compatible with
the sacrebleu reference implementation. The resampling itself is performed
in Rust for efficiency.

The resampling unit is the sentence. Each bootstrap iteration draws a
pseudo-test-set of the same size as the original by sampling sentences with
replacement, recomputes corpus-level BLEU for both systems on that sample,
and records which system performs better. The p-value is the proportion of
iterations in which the worse system appears to win.

Notes
-----
BLEU is a corpus-level metric and does not decompose meaningfully at the
sentence level. The sufficient statistics approach used here preserves
corpus-level correctness while avoiding redundant tokenisation on each
resample.

Tokenisation is handled by sacrebleu using the 13a tokeniser by default,
consistent with WMT evaluation practice. Scores produced by this module
are directly comparable to sacrebleu corpus-level BLEU scores computed
with the same settings.

For sentence-decomposable metrics such as chrF or COMET, use
:func:`evaluatio.hypothesis.paired_bootstrap_test` directly with
per-sentence scores.

References
----------
.. [1] Koehn, P. (2004). Statistical significance tests for machine
       translation evaluation. *Proceedings of EMNLP 2004*, 388-395.
.. [2] Post, M. (2018). A call for clarity in reporting BLEU scores.
       *Proceedings of the Third Conference on Machine Translation*, 186-191.
"""

from typing import Iterable, List

import sacrebleu
from evaluatio import _bindings


def _sentence_stats(
    hyp: str,
    refs: List[str],
    bleu: sacrebleu.BLEU,
) -> _bindings.BLEUSufficientStats:
    """
    Compute BLEU sufficient statistics for a single sentence pair.

    Parameters
    ----------
    hyp : str
        Hypothesis string.
    refs : list of str
        Reference strings. Must contain at least one non-empty reference.
    bleu : sacrebleu.BLEU
        Configured sacrebleu BLEU instance to use for scoring. Reused across
        sentences to avoid reinitialising tokenisation settings on every call.

    Returns
    -------
    BLEUSufficientStats
        Struct containing clipped n-gram match counts, total n-gram counts,
        hypothesis length, and reference length for this sentence.
    """
    s = bleu.sentence_score(hyp, refs)
    return _bindings.BLEUSufficientStats(s.counts, s.totals, s.sys_len, s.ref_len)


def bootstrap_bleu(
    references: Iterable[Iterable[str]],
    hyp1: Iterable[str],
    hyp2: Iterable[str],
    iterations: int,
    effective_order: bool = True,
) -> float:
    """
    Perform a paired bootstrap significance test comparing two MT systems
    using corpus-level BLEU.

    Sufficient statistics are precomputed per sentence using sacrebleu, then
    passed to the Rust resampling backend. On each bootstrap iteration a
    pseudo-test-set is drawn by sampling sentences with replacement, and
    corpus-level BLEU is recomputed for both systems from the accumulated
    sufficient statistics. The p-value is the proportion of iterations in
    which the worse system appears to outperform the better, using the
    ``(count + 1) / (iterations + 1)`` correction.

    Parameters
    ----------
    references : iterable of iterable of str
        Reference translations. The outer iterable has one entry per sentence,
        and the inner iterable contains one or more reference strings for that
        sentence. May be a generator; it is fully materialised before
        processing.
    hyp1 : iterable of str
        Hypothesis strings for the first system, one per sentence. Must be
        the same length as ``references``.
    hyp2 : iterable of str
        Hypothesis strings for the second system, one per sentence. Must be
        the same length as ``references``.
    iterations : int
        Number of bootstrap resamples. Values of 5000 to 10000 give stable
        p-value estimates for most purposes.
    effective_order : bool, optional
        If ``True`` (default), scales the n-gram order to the maximum order
        for which counts are non-zero. Recommended for sentence-level
        sufficient statistic computation to avoid zero precision on short
        segments. Set to ``False`` to match standard corpus-level BLEU
        behaviour exactly.

    Returns
    -------
    float
        Two-sided p-value in the range ``(0, 1]``. A value below 0.05
        indicates that the observed BLEU difference is unlikely under the
        null hypothesis that the two systems are equivalent.

    Raises
    ------
    ValueError
        If ``references``, ``hyp1``, and ``hyp2`` are not all the same
        length, or if any is empty.

    Notes
    -----
    The minimum possible p-value is ``1 / (iterations + 1)``. With
    ``iterations=9999`` this is 0.0001.

    Tokenisation uses sacrebleu's 13a tokeniser by default, consistent with
    WMT evaluation practice. BLEU scores computed internally are directly
    comparable to sacrebleu corpus-level scores produced with the same
    tokeniser and ``effective_order`` setting.

    This test does not account for variance across training runs. Two models
    differing only in random seed may yield a significant result. See
    Müller & Birch (2020) for discussion.

    Examples
    --------
    >>> references = [["the cat sat on the mat"], ["the dog ate the bone"]]
    >>> hyp1 = ["the cat sat on the mat", "the dog ate the bone"]
    >>> hyp2 = ["a cat sat on a mat", "a dog ate a bone"]
    >>> p = bootstrap_bleu(references, hyp1, hyp2, iterations=9999)
    >>> print(f"p = {p:.4f}")
    p = 0.0231
    """
    references = [list(r) for r in references]

    if len(references) == 0:
        raise ValueError("references must not be empty")

    hyp1 = list(hyp1)
    hyp2 = list(hyp2)

    if not (len(references) == len(hyp1) == len(hyp2)):
        raise ValueError(
            f"references, hyp1, and hyp2 must be the same length, "
            f"got {len(references)}, {len(hyp1)}, {len(hyp2)}"
        )

    bleu = sacrebleu.BLEU(effective_order=effective_order)

    stats_a = [_sentence_stats(h, r, bleu) for h, r in zip(hyp1, references)]
    stats_b = [_sentence_stats(h, r, bleu) for h, r in zip(hyp2, references)]

    return _bindings.bootstrap_bleu(stats_a, stats_b, iterations)