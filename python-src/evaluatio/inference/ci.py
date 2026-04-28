"""
Bootstrap-based confidence interval estimation utilities.

This module provides functionality to estimate empirical confidence
intervals using bootstrap resampling. It also defines a lightweight
container for returning confidence interval results.

The implementation relies on optimized native bindings for performance.
"""

from dataclasses import dataclass
from typing import Iterable

from evaluatio import _bindings


@dataclass
class ConfidenceInterval:
    """
    Container for confidence interval results.

    Attributes
    ----------
    mean : float
        The mean of the input sample.
    lower: float
        The lower bound of the confidence interval.
    upper: float
        The upper bound of the confidence interval.
    """

    mean: float
    lower: float
    upper: float


def bootstrap_confidence_interval(
    x: Iterable[float], iterations: int, alpha: float
) -> ConfidenceInterval:
    """
    Estimate a confidence interval using bootstrap resampling

    Parameters
    ----------
    x : Iterable[float]
        Input sample values.
    iterations : int
        Number of bootstrap resampling iterations.
    alpha : float
        Significance level for the confidence interval. For example,
        ``alpha=0.05`` corresponds to a 95% confidence interval.

    Returns
    -------
    ConfidenceInterval
        Results dataclass. Fields: ``mean`` (float with the mean of ``x``), ``lower`` (float, lower CI bound),
        ``upper`` (float, upper CI bound). See ConfidenceInterval for full documentation.

    Raises
    ------
    ValueError
        If ``iterations < 1``, if ``x`` is empty, or if ``alpha`` is not in (0, 1).

    Notes
    -----
    - The confidence interval is computed empirically via bootstrap sampling.
    - The returned mean is computed from the original sample, not the bootstrap distribution.

    Examples
    --------
    >>> from evaluatio.inference.ci import bootstrap_confidence_interval
    >>> values = [0.4, 0.7, 0.5, 0.9]
    >>> result = bootstrap_confidence_interval(values, 1000, 0.05)
    >>> result
    ConfidenceInterval(mean=0.625, lower=0.45, upper=0.8)
    """
    return _convert_confidence_interval(
        _bindings.bootstrap_confidence_interval(x, iterations, alpha)
    )


def error_rate_ci(
    counts: Iterable[int], exposure: Iterable[int], iterations: int, alpha: float
) -> ConfidenceInterval:
    """
    Estimate a confidence interval using bootstrap directly on error counts

    Parameters
    ----------
    counts : Iterable[int]
        The error counts
    exposure : Iterable[int]
        A list of lengths to normalise by
    iterations : int
        Number of bootstrap resampling iterations.
    alpha : float
        Significance level for the confidence interval. For example,
        ``alpha=0.05`` corresponds to a 95% confidence interval.

    Returns
    -------
    ConfidenceInterval
        Results dataclass. Fields: ``mean`` (float with the mean of ``x``), ``lower`` (float, lower CI bound),
        ``upper`` (float, upper CI bound). See ConfidenceInterval for full documentation.

    Raises
    ------
    ValueError
        If ``iterations < 1``, if ``x`` is empty, or if ``alpha`` is not in (0, 1).

    Notes
    -----
    - The confidence interval is computed empirically via bootstrap sampling of the counts and exposure.
    - The returned mean is computed from the original sample, not the bootstrap distribution.

    See Also
    --------
    - `word_error_rate_ci`, `character_error_rate_ci`, and `universal_error_rate_ci` are all wrappers
      around this function.

    Examples
    --------
    >>> from evaluatio.inference.ci import error_rate_confidence_interval
    >>> values = [0.4, 0.7, 0.5, 0.9]
    >>> result = error_rate_confidence_interval(values, 1000, 0.05)
    >>> result
    ConfidenceInterval(mean=0.625, lower=0.45, upper=0.8)
    """
    return _convert_confidence_interval(
        _bindings.error_rate_ci(counts, exposure, iterations, alpha)
    )


def _convert_confidence_interval(
    ci: _bindings.ConfidenceInterval,
) -> ConfidenceInterval:
    """Small helper function to convert from the Rust type to the Python type"""
    return ConfidenceInterval(mean=ci.mean, lower=ci.lower, upper=ci.upper)
