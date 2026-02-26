# Evaluatio

***Note: The library is under development, so things are likely to change, especially function signatures.***

Evaluatio is a library that contains computationally efficient metrics for the valuation of different NLP systems.
It is the continuation of the [`universal-edit-distance`](https://gitlab.com/prebens-phd-adventures/universal-edit-distance) project, but was renamed and restructured due to the project outgrowing its original purpose.

## Etymology
The name `evaluatio` is a Latin noun and means "evaluation". It also doubles as the English verb "to evaluate" with the Welsh verbal derivational suffix `-io`, so it could also be Welsh slang for "to evaluate".

## Components

### [evaluatio-core](./evaluatio-core/README.md)
`evaluatio-core` is a standalone Rust library that implements certain metrics and functions to improve the performance of the main library. This can be used by other Rust projects without requiring PyO3.

### [evaluatio-bindings](./evaluatio-bindings/README.md)
`evaluatio-bindings` contains the Py03 bindings between `evaluatio-core` and the main Python library. It simply exposes the functions and classes in the `evaluatio-core` to the Python library while also containing some helper functions to ensure that Python types are handled properly. All classes and functions are exported to a single Python module.

### python-src
`python-src` contains the main Python library. It contains wrappers for `evaluatio-bindings` to ensure that functions are documented, type-annotated, and sorted into different organised modules.

## Contribute to the project
There is always room for improvements, new metrics, new functionality, etc. If you have any suggestions or requests please feel free to add an issue! The main repository for the project can be found at [https://codeberg.org/prvinspace/evaluatio](codeberg.org/prvinspace/evaluatio)

## Maintainer

The project is maintained by Preben Vangberg &lt;prv21fgt@bangor.ac.uk&gt;.