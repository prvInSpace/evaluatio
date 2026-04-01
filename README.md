# Evaluatio
![PyPI - Version](https://img.shields.io/pypi/v/evaluatio)
![License](https://img.shields.io/github/license/prvInSpace/evaluatio)
[![CI](https://github.com/prvInSpace/evaluatio/actions/workflows/CI.yml/badge.svg)](https://github.com/prvInSpace/evaluatio/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/prvInSpace/evaluatio/graph/badge.svg?token=63NBX8175Q)](https://codecov.io/github/prvInSpace/evaluatio)

***Statistically rigorous evaluation for NLP with fast, correct metrics and built-in inference tools.***

## Why Evaluatio?
Most libraries make it easy to compute metrics.
Few make it easy to **evaluate models correctly**.

Evaluatio is designed to fix that.

It provides:
- **Correct metric implementations** (e.g. WER, CER)
- **Uncertainty estimation** (e.g. bootstrap confidence intervals)
- **Model comparison tools** (e.g. paired bootstrap tests)
- **Multiple testing correction** (e.g. Holm-Bonferroni)
- **High performance** (Rust-backed implementations)

## What's included

### Metrics
| Metric | Description |
|--------|-------------|
| WER / CER | Word and character error rate, including edit-distance variants |
| UED / UER | Type-agnostic edit distance/error rate. Works with any Python object supporting `__eq__` |
| PIER | Point of interest error rate, for evaluating punctuation, code-switching, capitalisation, etc. individually |

### Statistical inference
| Tool | Description |
|------|-------------|
| Bootstrap CI | Confidence intervals via bootstrap resampling (general, and corpus-correct variants for WER/CER) |
| Paired bootstrap test | Significance testing for model comparison |
| Multiple testing correction | Bonferroni and Holm-Bonferroni |
| Cohen's d | Effect size estimation (independent and paired) |

Current task guides cover **ASR**. The inference tools are general-purpose and apply across tasks.

## Quick example
```python
import pandas as pd
from evaluatio.metrics.wer import (
    word_error_rate,
    word_error_rate_ci,
    word_error_rate_per_pair,
)
from evaluatio.inference.bootstrap import paired_bootstrap_test

df = pd.read_csv("inferences.csv")

# Corpus-level WER
wer = word_error_rate(df["references"], df["predictions"])

# Confidence interval
ci = word_error_rate_ci(df["references"], df["predictions"], 5000, 0.95)

# Model comparison
wer_a = word_error_rate_per_pair(df["references"], df["model_a"])
wer_b = word_error_rate_per_pair(df["references"], df["model_b"])

pvalue = paired_bootstrap_test(wer_a, wer_b, iterations=5000)

print(f"WER: {ci.mean:.3f} ± {ci.upper - ci.mean:.3f}")
print(f"P-value: {pvalue}")
```

This workflow reflects the recommended approach: compute a corpus-level metric, estimate uncertainty, and compare models using paired statistical tests.

## Key features

### Correct by default
- Explicit, strongly typed APIs that reduce common evaluation mistakes.
- Documentation is designed as a reference for statistically rigorous evaluation, not just API usage. 

### Built-in statistical inference
- Bootstrap confidence intervals
- Paired bootstrap significance testing
- Multiple testing correction

### High performance
- Rust-backed core for efficient computation
- Faster WER/CER than common alternatives

## Installation
```bash
pip install evaluatio
```

## Documentation
Full documentation (including evaluation guides and statistical background): [https://prvinspace.github.io/evaluatio/](https://prvinspace.github.io/evaluatio/).

## Status
ASR metrics (WER, CER) and the statistical inference tools are stable. APIs elsewhere may change.

## Contribute to the project
There is always room for improvements, new metrics, new functionality, etc. If you have any suggestions or requests please feel free to add an issue! The main repository for the project can be found at [codeberg.org/prvinspace/evaluatio](https://codeberg.org/prvinspace/evaluatio).

## Maintainer
The project is maintained by Preben Vangberg &lt;[prv21fgt@bangor.ac.uk](mailto:prv21fgt@bangor.ac.uk)&gt;.

## Etymology
The name `evaluatio` is a Latin noun and means "evaluation". It also doubles as the English verb "to evaluate" with the Welsh verbal derivational suffix `-io`, so it could also be Welsh slang for "to evaluate".
