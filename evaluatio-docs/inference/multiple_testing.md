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

Holm-Bonferroni correction was introduced by Sturle Holm in 1979 [@holm1979correction]. It uses an iterative approach to ensure that FWER is at most $\alpha$. Assuming you have a sorted array of $p$-values $P_1, \dots, P_m$ for $m$ hypotheses, the formula for whether to reject hypothesis $H_k$ is given by:

$$
P_k \le \frac{\alpha}{m + 1 - k}
$$

It is less conservative than Bonferroni corrections, and is often preferred over it.

Holm-Bonferroni is implemented in Evaluatio as the function `holm_correction` in the package `evaluation.inference.multiple_testing`. It takes a list of $p$-values and returns an object containing the results of the correction and whether the null-hypothesis is rejected or not.