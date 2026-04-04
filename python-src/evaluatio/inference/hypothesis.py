"""
Hypothesis testing functions for paired samples.

This module provides paired statistical significance tests for comparing two
systems or models on matched observations. Both tests are non-parametric and
make no distributional assumptions, relying instead on resampling to construct
a null distribution.

The intended use case is evaluating whether the difference in performance
between two systems is statistically significant. For example two ASR models
evaluated on the same utterances, or two MT systems evaluated on the same source sentences.

All tests in this module are *paired*: every element ``x1[i]`` must correspond
to the same observation as ``x2[i]``. The ordering of pairs is assumed to be
meaningful and must be consistent between ``x1`` and ``x2``.

Notes
-----
For general guidance on which test to use:

- Use :func:`paired_bootstrap_test` when you want a p-value alongside
  a separately computed confidence interval, or when the bootstrap CI
  is your primary reported result.
- Use :func:`paired_permutation_test` when you want an exact significance
  test under the sharp null hypothesis of no effect on any unit.

Both tests converge to equivalent conclusions on large samples. On small
samples the permutation test has a slight power advantage due to its
exactness.
"""

from typing import Iterable

import evaluatio._bindings as _bindings


def paired_bootstrap_test(
    x1: Iterable[float],
    x2: Iterable[float],
    iterations: int,
) -> float:
    """
    Perform a paired bootstrap significance test on the mean difference.

    Resamples pairs with replacement to construct a null distribution of the
    mean difference under the hypothesis of no effect, and returns a two-sided
    p-value. The p-value is estimated as the proportion of bootstrap iterations
    in which the resampled mean difference is at least as extreme as the
    observed mean difference, using the ``(count + 1) / (iterations + 1)``
    correction to ensure the p-value is never exactly zero.

    Parameters
    ----------
    x1 : iterable of float
        Per-observation scores for the first system. Must be the same length
        as ``x2``, with ``x1[i]`` and ``x2[i]`` corresponding to the same
        observation.
    x2 : iterable of float
        Per-observation scores for the second system. Must be the same length
        as ``x1``.
    iterations : int
        Number of bootstrap resamples. Values of 5000 to 10000 give stable
        p-value estimates for most purposes. Larger values reduce Monte Carlo
        variance in the p-value but increase runtime linearly.

    Returns
    -------
    float
        Two-sided p-value in the range ``(0, 1]``. A value below 0.05
        indicates that the observed mean difference is unlikely under the null
        hypothesis of no effect.

    Raises
    ------
    ValueError
        If ``x1`` and ``x2`` have different lengths, or if either is empty.

    Notes
    -----
    The minimum possible p-value is ``1 / (iterations + 1)``. With
    ``iterations=9999`` this is 0.0001. Reporting p < 0.0001 without
    increasing ``iterations`` accordingly is not meaningful.

    This test resamples pairs with replacement, which models variability in
    the observed mean difference as if a different test set of the same size
    had been drawn. It does not account for variability across training runs
    or random seeds.

    Examples
    --------
    >>> x1 = [0.85, 0.90, 0.78, 0.92, 0.88]
    >>> x2 = [0.80, 0.85, 0.75, 0.88, 0.82]
    >>> p = paired_bootstrap_test(x1, x2, iterations=9999)
    >>> print(f"p = {p:.4f}")
    p = 0.0312
    """
    return _bindings.paired_bootstrap_test(x1, x2, iterations)


def paired_permutation_test(
    x1: Iterable[float],
    x2: Iterable[float],
    iterations: int,
    two_tailed: bool = True,
) -> float:
    """
    Perform a paired permutation significance test on the mean difference.

    Constructs a null distribution by randomly flipping the sign of each
    paired difference, under the sharp null hypothesis that the two systems
    are exchangeable on every observation. Returns a p-value estimated as the
    proportion of permutations producing a test statistic at least as extreme
    as the observed statistic, using the ``(count + 1) / (iterations + 1)``
    correction.

    Parameters
    ----------
    x1 : iterable of float
        Per-observation scores for the first system. Must be the same length
        as ``x2``, with ``x1[i]`` and ``x2[i]`` corresponding to the same
        observation.
    x2 : iterable of float
        Per-observation scores for the second system. Must be the same length
        as ``x1``.
    iterations : int
        Number of random permutations to sample. Values of 5000 to 10000 give
        stable p-value estimates for most purposes. The total number of
        distinct permutations for ``n`` pairs is ``2^n``, so exhaustive
        enumeration is only feasible for very small ``n``.
    two_tailed : bool, optional
        If ``True`` (default), the test is two-sided: both directions of
        difference contribute to the p-value. If ``False``, the test is
        one-sided in the direction where ``x1`` exceeds ``x2``.

    Returns
    -------
    float
        P-value in the range ``(0, 1]``. A value below 0.05 indicates that
        the observed mean difference is unlikely under the sharp null
        hypothesis of exchangeability.

    Raises
    ------
    ValueError
        If ``x1`` and ``x2`` have different lengths, if either is empty,
        or if the number of iterations is < 1.

    Notes
    -----
    The permutation test operates under a stricter null hypothesis than the
    bootstrap test: it assumes not merely that the mean difference is zero,
    but that the treatment assignment is completely arbitrary for every
    individual observation. This makes it more powerful than the bootstrap
    test on small samples, but the two converge on large samples.

    For a two-tailed test the sign-flip procedure is symmetric, so swapping
    ``x1`` and ``x2`` produces an identical p-value.

    The minimum possible p-value is ``1 / (iterations + 1)``.

    Examples
    --------
    Two-tailed test (default):

    >>> x1 = [0.85, 0.90, 0.78, 0.92, 0.88]
    >>> x2 = [0.80, 0.85, 0.75, 0.88, 0.82]
    >>> p = paired_permutation_test(x1, x2, iterations=9999)
    >>> print(f"p = {p:.4f}")
    p = 0.0287

    One-tailed test:

    >>> p = paired_permutation_test(x1, x2, iterations=9999, two_tailed=False)
    >>> print(f"p = {p:.4f}")
    p = 0.0144
    """
    return _bindings.paired_permutation_test(x1, x2, iterations, two_tailed)
