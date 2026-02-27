import evaluatio._bindings as _bindings


def cohens_d_independent(x1: list[float], x2: list[float]) -> float:
    """Calculates Cohen's d for two independent samples.

    NOTE: At least two samples is required in each array for the variances to make sense.

    NOTE: Should work with any type that is castable to a f64, hence ints, floats, and bools.
    """
    return _bindings.cohens_d_independent(x1, x2)


def cohens_d_paired(x1: list[float], x2: list[float]) -> float:
    """Calculates Cohen's d for two paired samples.

    Should be corrected using the formula d'/sqrt(1-r) after computation. 

    NOTE: At least two samples is required in each array for the variances to make sense.

    NOTE: Should work with any type that is castable to a f64, hence ints, floats, and bools.
    """
    return _bindings.cohens_d_paired(x1, x2)
