from typing import Any, Iterable

import evaluatio._bindings as _bindings


def poi_edit_distance(
    references: Iterable[Any], hypotheses: Iterable[Any], points_of_interest: Iterable[bool]
) -> Iterable[int]:
    """Calculates the edit distance between the two lists, but only on indicies where points_of_interest is set
    to True. points_of_interest has to be the same length as the reference."""
    return _bindings.poi_edit_distance(references, hypotheses, points_of_interest)


def poi_error_rate(
    references: Iterable[str], hypotheses: Iterable[str], points_of_interest: Iterable[bool]
) -> float:
    """Calculates the error distance between the two lists, but only on indicies where points_of_interest is set
    to True. points_of_interest has to be the same length as the reference."""
    return _bindings.poi_error_rate(references, hypotheses, points_of_interest)
