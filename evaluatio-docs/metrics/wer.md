# Word Error Rate (WER)

Word error rate is the most widely used metric for evaluating automatic speech recognition tasks. WER can best be explained as a length-normalised edit distance where the edit distance is defined as the number of substitutions, additions, and removals of words (or tokens) are required to turn the hypothesis into the reference. It is normalised based on the length of the reference. For a more comprehensive discussion about edit distance / error rate metrics, please see [Edit Distance](metrics/ued.md).

It is strongly related to other error rate based metrics such as [Character Error Rate (CER)](metrics/cer.md).

## Corpus level WER
In practice, the mean WER of a model is not computed as the simple average of test-level WERs. Instead, it is typically calculated as the total edit distance over the evaluation set divided by the total number of reference words. Formally, let $H = (H_1, \dots,H_n)$ be the hypotheses and $R=(R_1,\dots,R_n)$ the corresponding hypotheses. The corpus-level mean WER is defined as:

:::{math}
WER(H, R) = \frac{\sum_{i=1}^nedit\_distance(H_i, R_i)}{\sum_{i=1}^n|R_i|}.
:::

This differs from more conventional definition means, which for test-level WERs would be defined as:

:::{math}
WER(H, R) = \frac{\sum_{i=1}^nWER(H_i, R_i)}{|R|}
:::

## Evaluatio implemention
[API reference](api/metrics/wer.md)

The main Evaluatio implemention of WER is the `word_error_rate` function in `evaluatio.metrics.wer`. It is a wrapper around the type-agnostic error rate function [universal-error-rate](metrics/ued.md), but preprocesses the string beforehand by splitting them on whitespace. A per-pair is also provided called `word_error_rate_per_pair`.

If you wish to tokenize the strings using more complex tokenization methods, please pre-tokenize the strings and use the `universal-error-rate` function instead.

## Comparing WERs of different models

Because comparisons of WERs for models is an inherently paired problem and since inferences are paired, we can use a paired bootstrap test (`paired_bootstrap_test`) and the paired version of Cohen's $d$ (`paired_cohens_d`) to evaluate two models.

### Why conventional tests might not work
As with other length-normalised metrics, WERs are subject to certain distributional properties that complicate comparisons between different models.
The synopsis is that WERs are rarely normally distributed and the distribution of pairwise differences are regularly asymmetrical. As a consequence, regular statistical significance tests like Student's $t$-test and Wilcoxon signed-rank test are poorly suited when comparing ASR systems.

### Confidence interval
Because corpus-level WER is computed as a ratio of aggregated errors over aggregated reference tokens, uncertainty should be estimated via bootstrap resampling over utterances and recomputation of corpus-level WER. Confidence intervals derived from the distribution of utterance-level WERs do not correspond to the corpus-level metric and may misrepresent uncertainty.

### Multiple testing
When performing subgroup analyses (e.g., by gender, accent, age group), multiple statistical tests are often conducted simultaneously. Without correction, the probability of false positives increases.

The library does not automatically apply multiple comparison correction. Users may consider:

- Bonferroni correction (conservative)
- Holm correction
- False discovery rate control (Benjamini--Hochberg)

See the page about [multiple testing](inference/multiple_testing.md) for more info.

### Example evaluation code
```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from evaluatio.metrics.wer import word_error_rate_per_pair, word_error_rate_ci
model_1_wer_per_test = word_error_rate_per_pair(df["references"], df["model_1"])
model_2_wer_per_test = word_error_rate_per_pair(df["references"], df["model_2"])
model_1_ci = word_error_rate_ci(df["references"], df["model_1"], 5000, 0.95)
model_2_ci = word_error_rate_ci(df["references"], df["model_1"], 5000, 0.95)

from evaluatio.inference.bootstrap import paired_bootstrap_test
pvalue = paired_bootstrap_test(
    x1=model_1_wer_per_test,
    x2=model_2_wer_per_test,
    iterations=5000
)

from evaluatio.effect_size.cohen import cohens_d_paired
effect_size = cohens_d_paired(x1=model_1_wer_per_test, x2=model_2_wer_per_test)

def ci_plus_minus(ci):
    delta = ci.upper - ci.mean
    return f"{ci.mean:.3f} ± {delta:.3f}"

print(f"Model 1 WER: {ci_plus_minus(model_1_ci)}")
print(f"Model 2 WER: {ci_plus_minus(model_2_ci)}")
print(f"P-value: {pvalue}, Effect size: {effect_size}")
```

### Reporting recommendations
When reporting results for WERs comparisons, it is recommended to give the following:
- Mean WER for each model (and 95% CI)
- Paired bootstrap $p$-value 
- Effect size (paired Cohen's $d$)
