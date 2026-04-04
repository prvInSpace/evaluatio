"""
BLEU metrics

Evaluatio does not implement BLEU natively, but instead relies on `sacrebleu`<sup>3</sup>.
Evaluation complements `sacrebleu` by providing statistical comparison tools,
which are not included in `sacrebleu` itself. This module contains those
functions.

This module provides a paired bootstrap significance test for comparing two
machine translation systems using the BLEU metric. It follows the method
introduced by Koehn (2004), in which corpus-level BLEU is recomputed on
each bootstrap resample rather than aggregating per-sentence scores directly.

Sufficient statistics (clipped n-gram counts, total n-gram counts, and
lengths required for the brevity penalty) are precomputed per sentence using
`sacrebleu`, ensuring that tokenisation and scoring are fully compatible with
the `sacrebleu` reference implementation. The resampling itself is performed
in Rust for efficiency.

The confidence interval function works the same way as the paired bootstrap
and the required statistics are precomputed using `sacrebleu`.

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
.. [1] Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of
        machine translation. ACL.
.. [2] Koehn, P. (2004). Statistical significance tests for machine
       translation evaluation. *Proceedings of EMNLP 2004*, 388-395.
.. [3] Post, M. (2018). A call for clarity in reporting BLEU scores.
       *Proceedings of the Third Conference on Machine Translation*, 186-191.
"""

from typing import Iterable, List

from evaluatio.inference.ci import ConfidenceInterval, _convert_confidence_interval
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


def bleu_bootstrap_test(
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

    Examples
    --------
    >>> references = [["the cat sat on the mat"], ["the dog ate the bone"]]
    >>> hyp1 = ["the cat sat on the mat", "the dog ate the bone"]
    >>> hyp2 = ["a cat sat on a mat", "a dog ate a bone"]
    >>> p = bleu_bootstrap_test(references, hyp1, hyp2, iterations=9999)
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

    return _bindings.bleu_bootstrap_test(stats_a, stats_b, iterations)


def bleu_ci(
    references: Iterable[Iterable[str]],
    hypotheses: Iterable[str],
    iterations: int,
    alpha: float,
    effective_order: bool = True,
) -> ConfidenceInterval:
    """
    Estimate a confidence interval for corpus-level BLEU using bootstrap resampling.

    This function computes a percentile bootstrap confidence interval for the BLEU
    score by repeatedly resampling sentence-level sufficient statistics with
    replacement and recomputing corpus-level BLEU for each resample.

    The returned interval reflects uncertainty due to sampling variation in the
    evaluation dataset.

    Parameters
    ----------
    references : Iterable[Iterable[str]]
        Reference translations. Each element corresponds to a sample and should
        be an iterable of reference strings (to support multiple references per
        hypothesis).
    hypotheses : Iterable[str]
        Model predictions (hypotheses). Must be aligned with ``references`` such
        that each hypothesis corresponds to the same-indexed reference set.
    iterations : int
        Number of bootstrap resampling iterations. Larger values yield more stable
        estimates but increase computation time.
    alpha : float
        Significance level for the confidence interval. For example, ``alpha=0.05``
        corresponds to a 95% confidence interval.
    effective_order : bool, optional
        Whether to enable effective n-gram order when computing BLEU. This is
        passed directly to ``sacrebleu.BLEU`` and is recommended for shorter
        sequences. Default is ``True``.

    Returns
    -------
    ConfidenceInterval
        Estimated confidence interval for the corpus level word error rate.

    Notes
    -----
    - BLEU is not well-defined at the sentence level; this method operates on
      corpus-level BLEU computed from resampled sentence-level sufficient
      statistics.
    - The interval is computed using the percentile bootstrap method.
    - The resampling procedure preserves alignment between hypotheses and their
      corresponding references.
    - Small differences in BLEU may not be practically meaningful, even if
      confidence intervals do not overlap.

    References
    ----------
    .. [1] Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of
           machine translation. ACL.
    """

    bleu = sacrebleu.BLEU(effective_order=effective_order)

    stats = [_sentence_stats(h, list(r), bleu) for h, r in zip(hypotheses, references)]
    return _convert_confidence_interval(_bindings.bleu_ci(stats, iterations, alpha))
