# Edit Distance and Error Rate (UED/UER)

Edit distance measures the minimum number of substitutions, insertions, and deletions required to transform one sequence into another. It is also known as Levenshtein distance, after Vladimir Levenshtein who described the algorithm in 1966 [@levenshteinBinaryCodesCapable1966] in the context of correcting binary codes.

While edit distance is most commonly applied to characters or words in NLP, the algorithm itself only requires elements to be comparable — it makes no assumptions about what those elements are. In Python terms, any object implementing `__eq__` is sufficient. This makes it applicable to any tokenisation scheme, including custom or language-specific ones.

The normalised variant, universal error rate (UER), divides the edit distance by the length of the reference sequence:

$$UER(H, R) = \frac{\text{edit\_distance}(H, R)}{|R|}$$

[WER](/metrics/wer.md) and [CER](/metrics/cer.md) are both special cases of UER, differing only in how sequences are tokenised before the metric is computed.

## When to use UED/UER directly

In most cases you should use [WER](/metrics/wer.md) or [CER](/metrics/cer.md) directly. Use `universal_edit_distance_per_pair` or `universal_error_rate` when:

- You want to use a **custom tokenisation** scheme (e.g. language-specific segmentation, handling of contractions or hyphenated words)
- You are working with **non-string data** where elements support `__eq__`
- You need edit distance over **structured tokens** rather than raw characters or whitespace-delimited words

In these cases, pre-tokenise your sequences into lists and pass them directly to `universal_error_rate`. For corpus-level evaluation and confidence intervals over custom tokens, the same bootstrap tools used for WER and CER apply.

How to choose which function to use:
- Use `universal_error_rate` for a single corpus-level score.
- Use `universal_error_rate_per_pair` when you need utterance-level scores for downstream analysis (e.g. bootstrap tests, effect sizes).
- Use `universal_error_rate_ci` when you want uncertainty quantification on the corpus-level score directly.
- Use `universal_edit_distance_per_pair` when need to use the edit distances directly (e.g. for Poisson regression on error counts).

## Evaluatio implementation

[API reference](/api/metrics/uer.md)

The Rust implementation uses a generic function bounded by `PartialEq`, making it truly type-agnostic at the core level. The PyO3 bindings expose this to Python by implementing `PartialEq` for `PyAny` using the following dispatch:

- If both objects are the same native Rust type, comparison uses the native Rust implementation
- Otherwise, Python's `__eq__` is called directly

The second case carries a small performance overhead due to the Python call, but this only applies when comparing heterogeneous types, which should not occur in practice for well-formed inputs.

## Corpus-level UER

As with [WER](/metrics/wer.md) and [CER](/metrics/cer.md), corpus-level UER is computed as total edit distance divided by total reference length — not as a mean of utterance-level scores. The same distinction between micro and macro averaging applies here. See [WER](/metrics/wer.md) for a full discussion.
