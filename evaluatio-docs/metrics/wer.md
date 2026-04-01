# Word Error Rate (WER)

Word error rate is the most widely used metric for evaluating automatic speech recognition tasks. WER can best be explained as a length-normalised edit distance. The edit distance is defined as the number of substitutions, insertions, and removals of words (or tokens) required to turn the hypothesis into the reference. The result is normalised by the number of words in the reference. For a more comprehensive discussion about edit distance / error rate metrics, please see [Edit Distance](/metrics/ued.md).

It is strongly related to other error rate based metrics such as [Character Error Rate (CER)](/metrics/cer.md).

## Corpus level WER
In practice, the mean WER of a model is not computed as the simple average of utterance-level WERs. Instead, it is typically calculated as the total edit distance over the evaluation set divided by the total number of reference words. Formally, let $H = (H_1, \dots,H_n)$ be the hypotheses and $R=(R_1,\dots,R_n)$ the corresponding references. The corpus-level mean WER is defined as:

:::{math}
WER(H, R) = \frac{\sum_{i=1}^n \text{edit\_distance}(H_i, R_i)}{\sum_{i=1}^n|R_i|}.
:::

This differs from more conventional definition of mean, which for utterance-level WERs would be defined as:

:::{math}
WER_{\text{macro}}(H, R) = \frac{1}{n} \sum_{i=1}^n WER(H_i, R_i)
:::

The standard definition of WER used in ASR is the micro-average (corpus-level WER).
The macro-average (mean of utterance-level WERs) is a different quantity and is generally not reported, but is useful for statistical analysis.

## Limitations of WER
WER has many limitations that has been highlighted by various authors over the years. Some of these include:
- WER treats all errors equally regardless of semantic impact
- WER is sensitive to normalisation choices (casing, punctuation). See the [normalisation guide](/tasks/asr-evaluation#normalisation) for recommendations.
- WER can exceed 100%

Evaluatio resolves alignment deterministically, guaranteeing a unique and reproducible result. While multiple optimal alignments may exist, the total edit distance is invariant across them, hence the WER will always be correct.
This matters primarily for downstream analysis with [PIER](/metrics/pier.md), which is discussed on that page.

## Evaluatio implementation
[API reference](/api/metrics/wer.md)

The main Evaluatio implementation of WER is the `word_error_rate` function in `evaluatio.metrics.wer`. It is a wrapper around the type-agnostic error rate function [universal-error-rate](/metrics/ued.md), but preprocesses the string beforehand by splitting them on whitespace. While this is common for WER implementations, this assumes that whitespace tokenisation is appropriate for the language and task. Please ensure that it is appropriate for the language and task you are using it for.

A per-utterance variant is also provided: `word_error_rate_per_pair`.

How to choose which function to use:
- Use `word_error_rate` for a single corpus-level score.
- Use `word_error_rate_per_pair` when you need utterance-level scores for downstream analysis (e.g. bootstrap tests, effect sizes).
- Use `word_error_rate_ci` when you want uncertainty quantification on the corpus-level score directly.
- Use `word_edit_distance` when need to use the edit distances directly (e.g. for Poisson regression on error counts).

If you wish to tokenize the strings using more complex tokenization methods, please pre-tokenize the strings and use the `universal-error-rate` function instead.

## Comparing WERs of different models

Because comparisons of WERs for models is an inherently paired problem and since inferences are paired, we can use a paired bootstrap test (`paired_bootstrap_test`) to evaluate two models.
The paired bootstrap test operates over utterances, preserving the dependency structure between model outputs.

### Why conventional tests might not work
As with other length-normalised metrics, WERs are subject to certain distributional properties that complicate comparisons between different models.
The synopsis is that WERs are rarely normally distributed and the distribution of pairwise differences are regularly asymmetrical. As a consequence, regular statistical significance tests like Student's $t$-test and Wilcoxon signed-rank test are poorly suited when comparing ASR systems.

### Confidence interval
Because corpus-level WER is computed as a ratio of aggregated errors over aggregated reference tokens, uncertainty should be estimated via bootstrap resampling over utterances and recomputation of corpus-level WER. Confidence intervals derived from the distribution of utterance-level WERs do not correspond to the corpus-level metric and may misrepresent uncertainty. To make the process more efficient, a `word_error_rate_ci` function is also provided.

The CI returned by `word_error_rate_ci` is a bootstrap confidence interval on the corpus-level WER, i.e. the ratio of total errors to total reference tokens across the evaluation set. This is not a CI on the mean of utterance-level WERs, which would be a different quantity.

### Multiple testing
When performing subgroup analyses (e.g., by gender, accent, age group), multiple statistical tests are often conducted simultaneously. Without correction, the probability of false positives increases.

See the page about [multiple testing](/inference/multiple_testing.md) for more info.

### Example evaluation code
```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from evaluatio.metrics.wer import word_error_rate_per_pair, word_error_rate_ci
model_1_wer_per_test = word_error_rate_per_pair(df["references"], df["model_1"])
model_2_wer_per_test = word_error_rate_per_pair(df["references"], df["model_2"])
model_1_ci = word_error_rate_ci(df["references"], df["model_1"], 5000, 0.95)
model_2_ci = word_error_rate_ci(df["references"], df["model_2"], 5000, 0.95)

from evaluatio.inference.bootstrap import paired_bootstrap_test
pvalue = paired_bootstrap_test(
    x1=model_1_wer_per_test,
    x2=model_2_wer_per_test,
    iterations=5000
)

def ci_plus_minus(ci):
    delta = ci.upper - ci.mean
    return f"{ci.mean:.3f} ± {delta:.3f}"

print(f"Model 1 WER: {ci_plus_minus(model_1_ci)}")
print(f"Model 2 WER: {ci_plus_minus(model_2_ci)}")
print(f"P-value: {pvalue}")
```

### Reporting recommendations
When reporting results for WER comparisons, it is recommended to give the following:
- Mean WER for each model (and 95% CI)
- Paired bootstrap $p$-value 
- Absolute difference in WER

An example could look like:
> "Model A achieved a WER of 0.243 ± 0.008 (95% CI), compared to 0.31 ± 0.011 for Model B (paired bootstrap p=0.003), a reduction of 6.7 percentage points."
