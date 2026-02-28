# Character Error Rate (CER)

Character Error Rate (CER) is a length-normalised metric that measures the number of errors (additions, substitutions, and deletions required to change the hypothesis into the reference) and is normalised by the length of the reference.

It is strongly related to [](./wer.md) and [](./ued.md).

## Evaluatio implementation

The `character_error_rate` function is implemented as a simple wrapper around the [`universal_error_rate` function](metrics/ued.md), but strings are split into a list of UTF-8 characters before they are passed on to the function.
For more details, please see [](./ued.md)
