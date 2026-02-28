from typing import Iterable

import evaluatio._bindings as _bindings


def cohens_d_independent(x1: Iterable[float], x2: Iterable[float]) -> float:
    """Calculates Cohen's d for two independent samples.

    NOTE: At least two samples is required in each array for the variances to make sense.

    NOTE: Should work with any type that is castable to a f64, hence ints, floats, and bools.
    """
    return _bindings.cohens_d_independent(x1, x2)


def cohens_d_paired(x1: Iterable[float], x2: Iterable[float]) -> float:
    """Calculates Cohen's d for two paired samples.
    
    NOTE: At least two samples is required in each array for the variances to make sense.

    NOTE: Should work with any type that is castable to a f64, hence ints, floats, and bools.
    """
    return _bindings.cohens_d_paired(x1, x2)
