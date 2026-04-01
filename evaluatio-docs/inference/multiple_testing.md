# Multiple Testing

Correction is required when multiple tests are conducted on the same dataset to answer a single research question, for example, pairwise comparisons between $k$ models. It is generally not required when the same analysis is applied independently to multiple datasets or models as separate research questions, though conventions vary.

The reason is that the family-wise error rate (FWER) increases as the number of hypotheses increases. The good way to think of it is as the number comparisons we make the more likely we are to get a false positive.

Assuming that you want an overall $\alpha=0.05$ and you have $m=6$ hypotheses, the formula for the FWER is:

$FWER=1-(1-\alpha)^m=1-(1-0.05)^6\approx34\%$, rather than the desired 5%.

There are multiple ways of correcting for the increased family-wise error. In this article we will discuss the two most common ones. Note that in always all circumstances Holm-Bonferroni is preferred over regular Bonferroni correction.

## Methods

### Bonferroni correction

Bonferroni correction is an incredibly straightforward and easy way to correct for the multiple comparisons problem. Assuming that we have $m$ hypotheses we want to test (e.g. comparing 5 different dialect groups), we test each individual hypothesis at a significance level of $\alpha/m$.

For example, if we are testing $m = 5$ hypotheses with a desired $\alpha =0.05$, we would test each hypothesis at $\alpha=0.05/m = 0.01$.

Bonferroni correction is named after the Italian mathematician Carlo Emilio Bonferroni due to the methods use of the Bonferroni inequalities [@bonferroni1936teoria].

Bonferroni is implemented in Evaluatio as the function `bonferroni_correction` in the package `evaluation.inference.multiple_testing`. It takes a list of $p$-values and returns an object containing the results of the correction and whether the null-hypothesis is rejected or not.

### Holm-Bonferroni correction

Holm-Bonferroni correction was introduced by Sturle Holm in 1979 [@holm1979correction]. It uses an iterative approach to ensure that FWER is at most $\alpha$. Given $m$ hypotheses, sort the corresponding $p$-values in ascending order $P_1 \le P_2 \le \dots \le P_m$​. For each $k = 1, 2, \dots, m$ in order, reject $H_k$ if:

$$
P_k \le \frac{\alpha}{m + 1 - k}
$$

As soon as a hypothesis fails to be rejected, stop. All remaining hypotheses are retained. This sequential structure is what makes Holm-Bonferroni less conservative than Bonferroni correction, which applies the same threshold $\alpha/m$ to every hypothesis regardless of rank.
Bonferroni's conservatism leads to an increased Type II error rate (false negatives), meaning that you may fail to detect read differences. Holm-Bonferroni is therefore often preferred over it.

Holm-Bonferroni is implemented in Evaluatio as the function `holm_correction` in the package `evaluation.inference.multiple_testing`. It takes a list of $p$-values and returns an object containing the results of the correction and whether the null-hypothesis is rejected or not.

## Example code

The following code example compares 3 ASR models using [WER](/metrics/wer.md). That means that we have 3 combinations of models (`AB`, `AC`, and `BC`) that we need to test. Without correction this would lead to a $1−(1−0.05)^3=14\%$ false positive rate (Type I error) instead of the desired 5%. Holm-Bonferroni handles this for us.

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