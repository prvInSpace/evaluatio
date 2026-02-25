from typing import Any, List

import evaluatio._bindings as _bindings


def poi_edit_distance(
    predictions: List[Any], references: List[Any], points_of_interest: List[bool]
) -> List[int]:
    """Calculates the edit distance between the two lists, but only on indicies where points_of_interest is set
    to True. points_of_interest has to be the same length as the reference."""
    return _bindings.poi_edit_distance(predictions, references, points_of_interest)


def poi_error_rate(
    predictions: List[str], references: List[str], points_of_interest: List[bool]
) -> float:
    """Calculates the error distance between the two lists, but only on indicies where points_of_interest is set
    to True. points_of_interest has to be the same length as the reference."""
    return _bindings.poi_error_rate(predictions, references, points_of_interest)
