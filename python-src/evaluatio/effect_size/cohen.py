import evaluatio._bindings as _bindings


def cohens_d(x1: list[float], x2: list[float]) -> float:
    """Calculates Cohen's d for two given arrays.

    NOTE: At least two samples is required in each array for the variances to make sense.

    NOTE: Should work with any type that is castable to a f64, hence ints, floats, and bools.
    """
    return _bindings.cohens_d(x1, x2)
