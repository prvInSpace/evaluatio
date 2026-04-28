"""ChrF metric wrappers providing bootstrap confidence intervals and permutation tests.

This module provides thin wrappers around :class:`sacrebleu.CHRF` that add
statistical inference capabilities: bootstrap confidence intervals
(:func:`chrf_ci`) and paired permutation significance tests
(:func:`chrf_permutation_test`).

Both functions operate at the corpus level but delegate the underlying ChrF
arithmetic entirely to sacrebleu, ensuring scoring is identical to the
reference implementation.

Notes
-----
Sentence-level scores used in :func:`chrf_permutation_test` are obtained via
``sacrebleu.CHRF.sentence_score``; corpus-level scores and statistics used in
:func:`chrf_ci` are obtained via the internal ``_extract_corpus_statistics``
and ``_compute_score_from_stats`` helpers, which may change across sacrebleu
versions.
"""

from typing import Iterable, Sequence

import numpy as np
import sacrebleu

from evaluatio.inference.ci import ConfidenceInterval
from evaluatio.inference.hypothesis import paired_permutation_test


def chrf_ci(
    references: Iterable[Iterable[str]],
    hypotheses: Sequence[str],
    iterations: int,
    alpha: float,
    chrf: sacrebleu.CHRF | None = None,
    seed: int = 0,
) -> ConfidenceInterval:
    """Compute a bootstrap confidence interval for corpus ChrF score.

    Draws ``iterations`` bootstrap resamples (with replacement) from the
    per-sentence sufficient statistics, recomputes the corpus ChrF score for
    each resample, and returns the percentile-based confidence interval.

    Parameters
    ----------
    references : Iterable[Iterable[str]]
        Reference translations. Each element is an iterable of strings
        representing row in a test set. Has to be the same length as `hypotheses`.
        The length of all the elements has to be the same.
    hypotheses : Sequence[str]
        Model hypotheses, one string per sentence.
    iterations : int, optional
        Number of bootstrap resamples. Default is ``1000``. For
        publication-quality intervals, ``10_000`` or more is recommended.
    alpha : float, optional
        Significance level.  The returned interval covers
        ``1 - alpha`` of the bootstrap distribution. E.g. `0.05`` is 95 % CI.
    chrf : sacrebleu.CHRF or None, optional
        A pre-configured :class:`sacrebleu.CHRF` instance.  If *None*
        (default), a default instance is created with sacrebleu's standard
        parameters (``char_order=6``, ``word_order=0``, ``beta=2``).
    seed : int, optional
        Seed for the NumPy random number generator used during resampling,
        ensuring reproducibility.  Default is ``0``.

    Returns
    -------
    ConfidenceInterval

    Raises
    ------
    ValueError
        If the reference sets do not all have the same number of sentences.

    Examples
    --------
    >>> ci = chrf_ci([["the cat sat"]], ["the cat sat"])
    >>> ci.score
    100.0
    """
    references = [list(r) for r in references]
    if len(set(len(r) for r in references)) != 1:
        raise ValueError("All reference sets must have the same number of sentences.")

    if chrf is None:
        chrf = sacrebleu.CHRF()

    # sacrebleu expects references as a list of per-reference lists, where
    # each inner list contains one reference string per sentence.
    refs_sacrebleu = list(map(list, zip(*references)))

    # stats is a 2-D array of shape (n_sentences, n_statistics).  Each row
    # holds the sufficient statistics for one sentence, which sacrebleu can
    # aggregate to a corpus score via _compute_score_from_stats.

    stats = np.asarray(
        chrf._extract_corpus_statistics(hypotheses, refs_sacrebleu), dtype=np.float32
    )

    n_sentences, n_stats = stats.shape

    rng = np.random.default_rng(seed)

    # Accumulate bootstrap sums directly
    summed_stats = np.zeros((iterations, n_stats), dtype=np.float32)

    for _ in range(n_sentences):
        # sample which sentence index to pick for each bootstrap
        idx = rng.integers(0, n_sentences, size=iterations)
        summed_stats += stats[idx]

    # Now compute scores
    # The function takes a List[int] but handles np.ndarray without issue
    bootstrapped = [chrf._compute_score_from_stats(s).score for s in summed_stats]  # type: ignore
    lower, upper = np.percentile(bootstrapped, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return ConfidenceInterval(
        chrf.corpus_score(hypotheses, refs_sacrebleu).score,
        lower=float(lower),
        upper=float(upper),
    )


def chrf_permutation_test(
    references: Iterable[Iterable[str]],
    hypotheses1: Sequence[str],
    hypotheses2: Sequence[str],
    iterations: int,
    two_tailed: bool = True,
    chrf: sacrebleu.CHRF | None = None,
) -> float:
    """Run a paired approximate permutation test comparing two sets of hypotheses.

    For each sentence the ChrF score is computed independently for each
    hypothesis set.  The resulting paired score vectors are passed to
    :func:`~evaluatio.inference.hypothesis.paired_permutation_test`, which
    estimates the probability that the observed difference in means (or its
    absolute value, when ``two_tailed=True``) could arise by chance under the
    null hypothesis that the two systems are equivalent.

    Parameters
    ----------
    references : Iterable[Iterable[str]]
        Reference translations.  Each element is an iterable of strings
        representing one reference set.  All reference sets must contain one
        string per sentence and have the same length as the hypothesis
        sequences.
    hypotheses1 : Sequence[str]
        Hypothesis strings for system 1, one per sentence.
    hypotheses2 : Sequence[str]
        Hypothesis strings for system 2, one per sentence.
    iterations : int
        Number of permutation iterations.
    two_tailed : bool, optional
        If ``True`` (default), conduct a two-tailed test (H₁: systems differ).
        If ``False``, conduct a one-tailed test (H₁: system 1 is better).
    chrf : sacrebleu.CHRF or None, optional
        A pre-configured :class:`sacrebleu.CHRF` instance.  If *None*
        (default), a default instance is created.

    Returns
    -------
    float
        Estimated p-value.  Small values (e.g. < 0.05) indicate that the
        observed score difference is unlikely under the null hypothesis.

    Raises
    ------
    ValueError
        If the reference sets do not all have the same number of sentences.

    Examples
    --------
    >>> p = chrf_permutation_test(
    ...     [["the cat sat", "a dog ran"]],
    ...     ["the cat sat", "a dog ran"],
    ...     ["a cat sat", "the dog ran"],
    ... )
    >>> 0.0 <= p <= 1.0
    True
    """
    references = [list(r) for r in references]
    if len(set(len(r) for r in references)) != 1:
        raise ValueError("All reference sets must have the same number of sentences.")

    if chrf is None:
        chrf = sacrebleu.CHRF()

    hyp1_scores = [
        chrf.sentence_score(h, rs).score for rs, h in zip(references, hypotheses1)
    ]
    hyp2_scores = [
        chrf.sentence_score(h, rs).score for rs, h in zip(references, hypotheses2)
    ]

    return paired_permutation_test(hyp1_scores, hyp2_scores, iterations, two_tailed)
