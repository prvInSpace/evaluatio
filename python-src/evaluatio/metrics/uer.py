# Univeral functions
from typing import Any, List

import evaluatio._bindings as _bindings


def universal_error_rate_per_pair(
    references: List[List[str]], hypotheses: List[List[Any]]
) -> List[float]:
    """Calculates the error-rate for every zipped pair of references and hypotheses

    WARNING: This function is untested and you should sanity check the output of the function.

    NOTE: Due to limitations of pyo3 they only support a limited number of types. A new generic
    implementation is being tested that should allow a broader set of types as long as __eq__ is
    implemented.
    """
    return _bindings.universal_error_rate_per_pair(references, hypotheses)


def universal_edit_distance_per_pair(
    references: List[List[Any]], hypotheses: List[List[Any]]
) -> List[int]:
    """Calculates the edit-distance for every zipped pair of references and hypotheses

    WARNING: This function is untested and you should sanity check the output of the function.

    NOTE: Due to limitations of pyo3 they only support a limited number of types. A new generic
    implementation is being tested that should allow a broader set of types as long as __eq__ is
    implemented.
    """
    return _bindings.universal_edit_distance_per_pair(references, hypotheses)
