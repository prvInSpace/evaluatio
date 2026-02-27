"""Stub file for the _bindings package.

This should feature the same functions as those exported by evaluatio-bindings to aid development
of the Python package. No actual documentation for each file is needed since that is taken care off
at the wrapper level in the Python package itself.
"""

from typing import Any, List

# Univeral functions
def universal_error_rate_per_pair(
    references: List[List[str]], hypotheses: List[List[Any]]
) -> List[float]: ...
def universal_edit_distance_per_pair(
    references: List[List[str]], hypotheses: List[List[Any]]
) -> List[int]: ...

# Word-based helper functions
def word_error_rate_per_pair(
    references: List[str], hypotheses: List[str]
) -> List[float]: ...
def word_edit_distance_per_pair(
    references: List[str], hypotheses: List[str]
) -> List[int]: ...
def word_error_rate(references: List[str], hypotheses: List[str]) -> float: ...

# Character-based helper functions
def character_error_rate_per_pair(
    references: List[str], hypotheses: List[str]
) -> List[float]: ...
def character_edit_distance_per_pair(
    references: List[str], hypotheses: List[str]
) -> List[int]: ...
def character_error_rate(references: List[str], hypotheses: List[str]) -> float: ...

# Point of interest stuff
def poi_edit_distance(
    references: List[Any], hypotheses: List[Any], points_of_interest: List[bool]
) -> List[int]: ...
def poi_error_rate(
    references: List[str], hypotheses: List[str], points_of_interest: List[bool]
) -> float: ...

# Optimal alignment

class Alignment:
    index: int
    start: int
    end: int

def optimal_alignment(references: List[Any], hypotheses: List[Any]) -> List[Alignment]:
    """Returns a list of alignments for indices in one list to the indicies in the other.
    Note that this is only one of the optimal solutions as there can be multiple optimal alignments
    """

# Cohen's d
def cohens_d_independent(x1: List[float], x2: List[float]) -> float: ...
def cohens_d_paired(x1: List[float], x2: List[float]) -> float: ...
def paired_bootstrap_test(
    x1: List[float], x2: List[float], iterations: int
) -> float: ...
