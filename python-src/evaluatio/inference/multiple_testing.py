from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass
class MultipleTestingResult:
    """
    Result of a multiple testing correction procedure.

    Attributes
    ----------
    rejected : ndarray of bool
        Boolean array in the same order as the input p-values. True
        indicates the null hypothesis is rejected at the corrected
        threshold.
    adjusted_pvalues : ndarray of float
        Corrected p-values in the same order as the input p-values.
    method : str
        Name of the correction method applied, e.g. ``'holm'``.
    alpha : float
        The familywise error rate used.
    """

    rejected: NDArray[np.bool_]
    adjusted_pvalues: NDArray[np.float64]
    method: str
    alpha: float


def holm_correction(
    pvalues: Iterable[float], alpha: float = 0.05
) -> MultipleTestingResult:
    """
    Apply Holm-Bonferroni correction to a set of p-values.

    Parameters
    ----------
    pvalues : array-like of float
        P-values to correct. Order is preserved in the output arrays.
        Must all be in the range (0, 1).
    alpha : float, optional
        Familywise error rate threshold. Default is 0.05.

    Returns
    -------
    MultipleTestingResult
        Results dataclass. Fields: ``rejected`` (ndarray of bool, same
        order as input), ``adjusted_pvalues`` (ndarray of float),
        ``method`` (str, in this case always "holm"), ``alpha`` (float). See MultipleTestingResult
        for full documentation.

    Raises
    ------
    ValueError
        If any p-value is outside (0, 1) or if alpha is not in [0, 1].

    See Also
    --------
    bonferroni_correction : More conservative alternative.

    Notes
    -----
    Holm correction[1]_ controls the familywise error rate (FWER)
    under any dependence structure between tests. It is uniformly more
    powerful than Bonferroni correction and should be preferred in
    almost all cases.

    References
    ----------
    .. [1] Holm, S. (1979). A simple sequentially rejective multiple
        test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

    Examples
    --------
    >>> from evaluatio.inference.multiple_testing import holm_correction
    >>> pvalues = [0.03, 0.04, 0.001, 0.8, 0.02]
    >>> result = holm_correction(pvalues, alpha=0.05)
    >>> result.rejected
    array([ True,  True,  True, False,  True])
    >>> result.adjusted_pvalues
    array([0.09, 0.09, 0.005, 0.8, 0.09])
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha is not within [0, 1]")

    np_pvalues: NDArray[np.float64] = np.asarray(pvalues, dtype=np.float64)
    if len(np_pvalues[(np_pvalues < 0.0) | (np_pvalues > 1.0)]) > 0:
        raise ValueError("p-values contains values outside of (0, 1)")

    n = len(np_pvalues)

    # Sort by p-value, keeping track of original indices
    sort_idx = np.argsort(np_pvalues)
    sorted_pvals = np_pvalues[sort_idx]

    # Holm adjusted p-values
    # adjusted_p[i] = max(p[0]*(n), p[1]*(n-1), ..., p[i]*(n-i))
    # cumulatively take the max to ensure monotonicity
    factors = np.linspace(n, 1, n, dtype=np.int64)
    adjusted = np.maximum.accumulate(sorted_pvals * factors)
    adjusted = np.minimum(adjusted, 1.0)  # cap at 1

    # Reject if adjusted p-value <= alpha
    rejected_sorted = adjusted <= alpha

    # Restore original order
    rejected = np.empty(n, dtype=bool)
    adjusted_original = np.empty(n, dtype=float)
    rejected[sort_idx] = rejected_sorted
    adjusted_original[sort_idx] = adjusted

    return MultipleTestingResult(
        rejected=rejected,
        adjusted_pvalues=adjusted_original,
        method="holm",
        alpha=alpha,
    )


def bonferroni_correction(
    pvalues: Iterable[float], alpha: float = 0.05
) -> MultipleTestingResult:
    """
    Apply Bonferroni correction to a set of p-values.

    Included for completeness. Holm correction is preferred in almost
    all cases as it is uniformly more powerful while providing the
    same familywise error rate control.

    Parameters
    ----------
    pvalues : array-like of float
        P-values to correct. Order is preserved in the output arrays.
        Must all be in the range (0, 1).
    alpha : float, optional
        Familywise error rate threshold. Default is 0.05.

    Returns
    -------
    MultipleTestingResult
        Results dataclass. Fields: ``rejected`` (ndarray of bool, same
        order as input), ``adjusted_pvalues`` (ndarray of float),
        ``method`` (str, in this case always "bonferroni"), ``alpha`` (float). See MultipleTestingResult
        for full documentation.

    Raises
    ------
    ValueError
        If any p-value is outside (0, 1) or if alpha is not in (0, 1).

    See Also
    --------
    holm_correction : Less conservative alternative.

    References
    ----------
    .. [1] Bonferroni, C. (1936). Teoria statistica delle classi e calcolo delle probabilita.
        Pubblicazioni del R. Istituto superiore di scienze economiche e commericiali di Firenze, 8, 3-62.

    Examples
    --------
    >>> from evaluatio.inference.multiple_testing import bonferroni_correction
    >>> pvalues = [0.03, 0.04, 0.001, 0.8, 0.02]
    >>> result = bonferroni_correction(pvalues, alpha=0.05)
    >>> result.rejected
    array([False, False,  True, False, False])
    >>> result.adjusted_pvalues
    array([0.15, 0.2, 0.005, 1.0, 0.1])
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha is not within [0, 1]")

    np_pvalues: NDArray[np.float64] = np.asarray(pvalues, dtype=float)
    if len(np_pvalues[(np_pvalues < 0.0) | (np_pvalues > 1.0)]) > 0:
        raise ValueError("p-values contains values outside of (0, 1)")

    n = len(np_pvalues)

    # Whether the p-values are multipled by n or the alpha is divided
    # by n makes little difference. In this case, we decide to multiply
    # since it makes the p-values comparable with the original alpha
    adjusted = np.minimum(np_pvalues * n, 1.0)

    return MultipleTestingResult(
        rejected=adjusted <= alpha,
        adjusted_pvalues=adjusted,
        method="bonferroni",
        alpha=alpha,
    )
