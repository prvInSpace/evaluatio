# Automatic Speech Recognition Evaluation

Welcome to this guide on how to evaluate automatic speech recognition (ASR) systems comprehensively and using proper statistical methods.

## Choosing a metric

There are a handful of metrics to be aware of:

| Metrics | Details |
| --- | --- |
[WER](/metrics/wer.md) | Word error rate (WER) is the most commonly used metric in ASR. It quantifies the number of word level errors normalised by the number of words in the reference string. Errors are defined as the number of substitutions, insertions, and deletions required to change the hypothesis to the reference.
[CER](/metrics/cer.md) | Character error rate (CER) is the same as WER but on a character level. Commonly used in ASR literature and especially useful for logographic languages.

Since [WER](/metrics/wer.md) is the most commonly used metric in ASR, this guide will use that as an example.

## Corpus-level vs utterance level evaluation

Certain metrics (e.g. [WER](/metrics/wer.md) and [CER](/metrics/cer.md)) calculate the summary corpus level statistic differently than a simple mean of the utterance level results. Corpus level statistics are the standard reported metrics. It is important to note that a mean of utterance-level metrics **does not always equal** the corpus-level statistic (e.g. see [WER](/metrics/wer.md)).

Confidence intervals are typically computed using bootstrap resampling over utterances, where each resample recomputes the corpus-level WER.

Utterance-level statistics are also used in some circumstances. For example when performing statistical testing, comparing different models, bootstrapping, estimation of variance etc. Utterance-level statistics are never reported as the main metric.

## Normalisation

When evaluating ASR models, it is important that we normalise both reference and prediction strings in a way that makes them comparable. `hello`, `hello.`, and `Hello` are all different and as such would often be flagged as mistakes by our system. In general, there are a handful of decisions that you should make before evaluating the models:

- **Do we care about punctuation?** If not, then remove them from *both* the reference and the prediction strings. If we do care about punctuation, then we likely don't want to treat them as part of the word in front. This can be resolved by adding a space between punctuation marks and words or tokenizing the strings manually. If strings are tokenized manually, you should use [UER](/metrics/uer.md) instead of [WER](/metrics/wer.md).
- **Do we care about case?** If not, then normalise the case for both references and hypotheses.
- **Do we care about tags?** Sometimes references contains tags to indicate certain special sounds such as laughter. If the system is trained to handle these then it might be reasonable to leave them in. Otherwise, you should remove them from the references.
- **How are words defined?** WER commonly tokenize strings based on whitespace. This might not be optimal for all languages. Handling contractions, hyphenated words, or language-specific segmentation directly affect WER and must be consistent across systems. 

In certain circumstances, it might be desirable to evaluate things like punctuation, capitalisation, etc. individually. There are various ways of doing this, but [PIER](/metrics/pier.md) can be used for this purpose.


## Single model evaluation

### Example code

```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from evaluatio.metrics.wer import word_error_rate, word_error_rate_ci
model_wer = word_error_rate(df["references"], df["predictions"])
model_ci = word_error_rate_ci(df["references"], df["predictions"], 5000, 0.95)

print(f"Model WER: {model_wer}")
print(f"Model CI:  {model_ci.mean:.3f} ± {model_ci.upper-model_ci.mean:.3f}")
```

### Example reporting
If you prefer precentages:
> The model achieved a WER of 24.3% (95% CI: 23.5%-25.1%) on the test set.

Otherwise:
> The model achieved a WER of 0.243 (95% CI: 0.235–0.251) on the test set.

## Comparing two models

Comparing two models is normally quite straightforward, but there are quite a few pitfalls that you can fall into.

### Ensuring models are tested on the same data
Models should _always_ be tested on the same utterances. This ensures that the comparison is valid, and we are comparing apples with apples. When testing on different test sets the results are much less reliable since it introduces many more variables that we cannot account for.

### Choosing an appropriate statistical test
Most metrics used for ASR have distributional properties that make various statistical inference tests unreliable for ASR purposes.
Metrics such as [WER](/metrics/wer.md) have their own section about this on their page, so please refer to the page for the metric that you are using.
In general, tests such as $t$-tests, Wilcoxon signed-rank test, etc. are not reliable for edit distance based metrics such as [WER](/metrics/wer.md) and [CER](/metrics/cer.md).
This is because edit-distance metrics like [WER](/metrics/wer.md) are non-linear, bounded, and highly non-normal, violating assumptions of these tests.
Paired bootstrap test has been shown to be generally reliable and should be used over most other alternatives.

### Example code
```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from evaluatio.metrics.wer import word_error_rate_per_pair, word_error_rate_ci
from evaluatio.inference.bootstrap import paired_bootstrap_test

model_1_wer_per_test = word_error_rate_per_pair(df["references"], df["model_1"])
model_2_wer_per_test = word_error_rate_per_pair(df["references"], df["model_2"])
model_1_ci = word_error_rate_ci(df["references"], df["model_1"], 5000, 0.95)
model_2_ci = word_error_rate_ci(df["references"], df["model_2"], 5000, 0.95)

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

## Multiple testing
See separate page on [multiple testing](/inference/multiple_testing.md) for more details.

When running multiple comparisons, be that testing multiple models (3 models or more), comparing multiple dialects, etc., you should always correct for multiple testing. This is because when you are testing multiple hypotheses, the familywise error rate (FWER) becomes higher than the alpha leading to unreliable results. E.g. if you are testing 6 different dialects, the FWER is actually $1−(1−0.05)^6=34\%$ rather than the desired alpha of 5%.

For most ASR applications, [Holm-Bonferroni](/inference/multiple-testing.md#holm-bonferroni-correction) is preferred over regular [Bonferroni correction](/inference/multiple-testing#bonferroni-correction).

### Example code
The following code example compares 3 models. That means that we have 3 combinations of models (`AB`, `AC`, and `BC`) that we need to test. Without correction this would lead to a $1−(1−0.05)^3=14\%$ false positive rate (Type I error) instead of the desired 5%. Holm-Bonferroni handles this for us.

```python
import pandas as pd
import itertools
import numpy as np

from evaluatio.metrics.wer import word_error_rate_per_pair
from evaluatio.inference.bootstrap import paired_bootstrap_test
from evaluatio.inference.multiple_testing import holm_correction

# This assumes inferences are already preprocessed and normalised
df = pd.read_csv("inferences.csv")
references = df["references"]
models = {
    "model_a": df["model_a"],
    "model_b": df["model_b"],
    "model_c": df["model_c"],
}

# Compute per-pair WERs for each model
wer_per_pair = {
    name: word_error_rate_per_pair(references, hypotheses)
    for name, hypotheses in models.items()
}

# Run pairwise bootstrap tests for all model combinations
model_names = list(models.keys())
pairs = list(itertools.combinations(model_names, 2))

pvalues = np.array([
    paired_bootstrap_test(
        x1=wer_per_pair[a],
        x2=wer_per_pair[b],
        iterations=5000
    )
    for a, b in pairs
])

# Apply Holm correction for multiple comparisons
correction = holm_correction(pvalues, alpha=0.05)

# Report results
for (a, b), pval, adj_pval, rejected in zip(
    pairs, pvalues, correction.adjusted_pvalues, correction.rejected
):
    print(f"{a} vs {b}: p={pval:.4f}, p_adj={adj_pval:.4f}, significant={rejected}")
```