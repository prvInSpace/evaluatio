"""Stub file for the _bindings package.

This should feature the same functions as those exported by evaluatio-bindings to aid development
of the Python package. No actual documentation for each file is needed since that is taken care off
at the wrapper level in the Python package itself.
"""

from typing import Any, Iterable

# Univeral functions
def universal_error_rate_per_pair(
    references: Iterable[Iterable[str]], hypotheses: Iterable[Iterable[Any]]
) -> Iterable[float]: ...
def universal_edit_distance_per_pair(
    references: Iterable[Iterable[str]], hypotheses: Iterable[Iterable[Any]]
) -> Iterable[int]: ...

# Word-based helper functions
def word_error_rate_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> Iterable[float]: ...
def word_edit_distance_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> Iterable[int]: ...
def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float: ...

# Character-based helper functions
def character_error_rate_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> Iterable[float]: ...
def character_edit_distance_per_pair(
    references: Iterable[str], hypotheses: Iterable[str]
) -> Iterable[int]: ...
def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float: ...

# Point of interest stuff
def poi_edit_distance(
    references: Iterable[Any], hypotheses: Iterable[Any], points_of_interest: Iterable[bool]
) -> Iterable[int]: ...
def poi_error_rate(
    references: Iterable[str], hypotheses: Iterable[str], points_of_interest: Iterable[bool]
) -> float: ...

# Optimal alignment

class Alignment:
    index: int
    start: int
    end: int

def optimal_alignment(references: Iterable[Any], hypotheses: Iterable[Any]) -> Iterable[Alignment]:
    """Returns a list of alignments for indices in one list to the indicies in the other.
    Note that this is only one of the optimal solutions as there can be multiple optimal alignments
    """

# Cohen's d
def cohens_d_independent(x1: Iterable[float], x2: Iterable[float]) -> float: ...
def cohens_d_paired(x1: Iterable[float], x2: Iterable[float]) -> float: ...
def paired_bootstrap_test(
    x1: Iterable[float], x2: Iterable[float], iterations: int
) -> float: ...

class ConfidenceInterval:
    mean: float
    lower: float
    upper: float

def confidence_interval(
    x: Iterable[float], iterations: int, alpha: float
) -> ConfidenceInterval: ...
def word_error_rate_ci(
    references: Iterable[str], hypotheses: Iterable[str], iterations: int, alpha: float
) -> ConfidenceInterval: ...
