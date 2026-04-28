# Bilingual Evaluation Understudy (BLEU)

BLEU is a metric commonly used in machine translation [@papineni2002bleu].

## Limitations of BLEU
- BLEU is insensitive to meaning and semantic adequacy.
- BLEU scores are not comparable across different tokenisations.
- BLEU is not well-defined on a sentence level.

## Evaluatio implementation
[API reference](/api/metrics/bleu)

Evaluatio does not implement BLEU natively, but instead relies on [`sacrebleu`](https://github.com/mjpost/sacrebleu) [@post2018sacrebleu]. This is to preserve reproducibility and tokenisation standardisation.
Evaluatio complements `sacrebleu` by providing statistical comparison tools, which are not included in `sacrebleu` itself.

The two functions that are available are:
- `bleu_bootstrap_test` which is used to compare two models.
- `bleu_ci` which is used to calculate a confidence interval for a single model.

## Comparing BLEU scores of different models

BLEU is not well-defined at the sentence level due to its reliance on n-gram precision and brevity penalty.
Sentence-level BLEU scores are highly variable and often zero, making them unsuitable for statistical tests that assume stable per-sample measurements.

As a result, tests such as paired permutation or paired bootstrap over sentence-level scores are unreliable for BLEU.

@koehn2004statistical introduced a method of comparing BLEU scores of two models using bootstrapping.
The method repeatedly samples sentences with replacement to form new pseudo-corpora.
For each resampled pseudo-corpus, corpus-level BLEU is computed for both models.

The $p$-value is the proportion of resamples in which the inferior model, as determined on the original test set, matches or outperforms the superior model, estimated as $\frac{\text{count} + 1}{N + 1}$​ where $N$ is the number of iterations. A small $p$-value indicates that the observed BLEU advantage is unlikely to be due to test set sampling variability alone.

The resampling is performed on aligned (reference, hypothesis<sub>1</sub>, hypothesis<sub>2</sub>) triples, preserving the pairing between model outputs.

### Example code

**N.B:** BLEU expects a nested list of reference strings since each row can have more than one reference string. This example code assumes that `df["references"]` a nested list of strings, so make sure that your dataset actually follows this format.

```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from sacrebleu import BLEU
bleu = BLEU(effective_order=True)

model_1 = bleu.corpus_score(df["model_1"], df["references"])
model_2 = bleu.corpus_score(df["model_2"], df["references"])

from evaluatio.metrics.bleu import bleu_bootstrap_test, bleu_ci

model_1_ci = bleu_ci(df["references"], df["model_1"], 999, 0.05)
model_2_ci = bleu_ci(df["references"], df["model_2"], 999, 0.05)

pvalue = bleu_bootstrap_test(
    df["references"],
    df["model_1"],
    df["model_2"],
    iterations=9999
)

print(f"Model 1 BLEU: {model_1.score} ± {model_1_ci.upper - model_1_ci.mean}")
print(f"Model 2 BLEU: {model_2.score} ± {model_2_ci.upper - model_2_ci.mean}")
print(f"P-value: {pvalue}")
```

### Reporting and interpreting the result

The $p$-value and confidence interval answer different questions and should be reported together.

The $p$-value from `bleu_bootstrap_test` asks what the likelihood is that the difference is caused by random chance: a value below 0.05 provides evidence that the performance gap is unlikely to be explained by test set sampling variability alone. A value at or above 0.05 suggests the observed difference may be within the range of chance variation for a test set of this size.

The confidence interval from `bleu_ci` asks how large the BLEU score of a single model is, and how precisely the test set estimates it. A narrow CI indicates a stable, reliable estimate; a wide CI indicates that a different test set of the same size might yield a noticeably different score.

A statistically significant result with overlapping CIs is entirely possible since the comparison test and the single-model CI are measuring different things.
The bootstrapped $p$-value from the hypothesis test is the appropriate tool for deciding whether one model is likely better than another; the CI is the appropriate tool for reporting how good each model is in absolute terms.

