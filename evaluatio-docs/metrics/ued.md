# Edit Distance (ED/UED)

Edit distance a metric that measures the number of substitutions, additions, and removals are required to change one array into another. It is also often called the Levenshtein distance due to the person who invented it.

Note that I said array, not a specific type, and that was intentional. The original algorithm described by Levenshtein was correcting binary information [@levenshteinBinaryCodesCapable1966], but nowadays it is often used in NLP to measure character or world level errors (see [](./cer.md) and [](./wer.md)). Despite this, the algorithm only requires elements to be comparable (i.e. they implement `__eq__` in Python). This makes the algorithm very versatile, *universal* if you will.

## Evaluatio implementation

The implementation of a type agnostic version of the Levenshtein distance algorithm in Rust is trivial. Everything that is required is a generic function that takes a type `T`, that is bounded by `PartialEq`.

In order to make PyO3 work with the Rust function, we need to implement `PartialEq` for the `PyAny` class.
The way it is implemented in Evaluatio if:
- if the types match then they are compared using their primitive Rust types, or
- if not then a call the Python objects `__eq__` function is dispatched.
Note that the call to `__eq__` will be slower than the using the native comparisons.

### Metrics that are based on UED
- [](./cer.md)
- [](./wer.md)
