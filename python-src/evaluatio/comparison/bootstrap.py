from typing import List

import evaluatio._bindings as _bindings


def paired_bootstrap_test(x1: List[float], x2: List[float], iterations: int) -> float:
    """Performs a paired bootstrap significance test on the mean difference
    between two paired samples, returning a two-sided p-value.

    NOTE: This is a paired test: every element `x1[i]` must correspond to `x2[i]`
    """
    return _bindings.paired_bootstrap_test(x1, x2, iterations)
