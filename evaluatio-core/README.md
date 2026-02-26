# evaluatio-core

This component contains a standalone Rust library that implements certain metrics and functions to improve the performance of the main library. This can be used by other Rust projects without requiring PyO3.

PyO3 is needed to wrap structs with `pyclass`, however, this is disabled by default. It can be enabled by passing the features flag `python`.
