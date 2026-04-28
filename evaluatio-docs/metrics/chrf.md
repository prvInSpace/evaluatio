# Character F-Score (ChrF)

ChrF is an $n$-gram based evaluation metric for machine translation that operates at the character level rather than the word level. It is defined as the F-score of character $n$-gram precision and recall between a hypothesis and one or more references. The use of character $n$-grams makes ChrF more robust to morphological variation than word-level metrics such as [BLEU](/metrics/bleu.md), and generally correlates better with human judgements, particularly for morphologically rich languages.

The standard variant used in practice is ChrF++ (``word_order=2``), which augments character $n$-grams with word unigrams and bigrams. Unless otherwise stated, references to ChrF on this page apply to both variants.

## Corpus-level ChrF

Like [WER](/metrics/wer.md), corpus-level ChrF is not the simple average of sentence-level scores. It is computed from aggregated character $n$-gram counts across the entire evaluation set: precision and recall are computed over the totals, and the F-score is derived from those aggregates. This means the corpus-level score cannot be recovered by averaging sentence-level ChrF scores.

This has an important consequence for uncertainty estimation: confidence intervals must be estimated by resampling over sentences and recomputing the corpus-level score from scratch each time, not by resampling the distribution of sentence-level scores. See Confidence interval below.

## Evaluatio implementation

[API reference](/api/metrics/chrf)

Evaluatio does not implement ChrF natively, but instead relies on [`sacrebleu`](https://github.com/mjpost/sacrebleu) [@post2018sacrebleu]. Evaluatio complements `sacrebleu` by providing statistical comparison tools, which are not included in `sacrebleu` itself.

In contrast to [BLEU](/metrics/bleu) and [WER](/metrics/wer), there are no metric-specific functions for ChrF in Evaluatio. Instead, two thin wrappers are provided in `evaluatio.metrics.chrf` that bridge `sacrebleu`'s scoring with Evaluatio's inference functions:

- Use `chrf_ci` when you want a bootstrap confidence interval on the corpus-level ChrF score.
- Use `chrf_permutation_test` when comparing two models with a paired permutation test.

For the permutation test, sentence-level ChrF scores are well-behaved enough that the generic `paired_bootstrap_test` or `permutation_test` can also be used directly on sentence-level scores obtained from `sacrebleu.CHRF.sentence_score`. The dedicated wrapper is provided for convenience and a consistent interface.

If you need sentence-level ChrF scores for other downstream analysis, obtain them directly from `sacrebleu`:

```python
import sacrebleu
chrf = sacrebleu.CHRF()
sentence_scores = [
    chrf.sentence_score(hyp, refs).score
    for hyp, refs in zip(hypotheses, references)
]
```

## Comparing ChrF scores of different models

Compared to [BLEU](/metrics/bleu.md), ChrF is fairly well-behaved at the sentence level. Sentence-level scores are bounded between 0 and 100, are not as heavily zero-inflated as sentence BLEU, and pairwise differences tend to have a more symmetric distribution. This means that a paired permutation test or paired bootstrap test over sentence-level scores works well in practice.

### Why conventional tests might not work

As with other corpus-level metrics, ChrF scores are subject to distributional properties that complicate direct model comparisons. Sentence-level ChrF differences are not normally distributed, so Student's $t$-test is generally inappropriate. The paired permutation test makes no distributional assumptions and is the recommended approach.

### Confidence interval

Because corpus-level ChrF is computed from aggregated $n$-gram counts rather than as a mean of sentence scores, confidence intervals must be estimated by bootstrapping over sentences and recomputing the full corpus-level score each time. The `chrf_ci` function handles this correctly using `sacrebleu`'s internal sufficient statistics.

Confidence intervals derived from the distribution of sentence-level ChrF scores do not correspond to the corpus-level metric and may misrepresent uncertainty. Use `chrf_ci` rather than the generic `bootstrap_ci` for this reason.

### Example evaluation code

```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from evaluatio.metrics.chrf import chrf_ci, chrf_permutation_test

# List of references per row. Note that this is different from
# the format required by sacrebleu's corpus statistic. It is the
# same as what sentence_score uses. All the nested lists should
# have the same length
references = [[r] for r in df["references"]]

model_1_ci = chrf_ci(references, list(df["model_1"]), iterations=5000, alpha=0.05)
model_2_ci = chrf_ci(references, list(df["model_2"]), iterations=5000, alpha=0.05)

pvalue = chrf_permutation_test(
    references=references,
    hypotheses1=list(df["model_1"]),
    hypotheses2=list(df["model_2"]),
    iterations=5000,
)

def ci_plus_minus(ci):
    delta = ci.upper - ci.mean
    return f"{ci.mean:.3f} ± {delta:.3f}"

print(f"Model 1 ChrF: {ci_plus_minus(model_1_ci)}")
print(f"Model 2 ChrF: {ci_plus_minus(model_2_ci)}")
print(f"P-value: {pvalue}")
```

If you prefer a bootstrap test over a permutation test, you can obtain sentence-level scores from `sacrebleu` directly and pass them to the generic `paired_bootstrap_test`:

```python
import sacrebleu

from evaluatio.inference.hypothesis import paired_bootstrap_test

chrf = sacrebleu.CHRF()
model_1_scores = [
    chrf.sentence_score(h, rs).score
    for h, rs in zip(df["model_1"], references)
]
model_2_scores = [
    chrf.sentence_score(h, rs).score
    for h, rs in zip(df["model_2"], references)
]

pvalue = paired_bootstrap_test(model_1_scores, model_2_scores, iterations=5000)
```

Note that this tests for a difference in mean sentence-level ChrF, which is a subtly different quantity from the difference in corpus-level ChrF. For most practical purposes the conclusions will agree, but `chrf_permutation_test` is preferred when the corpus-level score is the quantity of interest.

### Multiple testing

When performing subgroup analyses (e.g. by language pair, domain, or document type), multiple statistical tests are often conducted simultaneously. Without correction, the probability of false positives increases.

See the page about [multiple testing](/inference/multiple_testing.md) for more info.

### Reporting recommendations

When reporting results for ChrF comparisons, it is recommended to give the following:

- Corpus-level ChrF score for each model (and 95% CI)
- Paired permutation test $p$-value
- Absolute difference in ChrF

An example could look like:

> "Model A achieved a ChrF score of 61.3 ± 0.8 (95% CI), compared to 58.7 ± 0.9 for Model B (paired permutation p=0.004), an absolute improvement of 2.6 points."

## Limitations of ChrF

- ChrF is sensitive to the choice of character $n$-gram order and the $\beta$ parameter controlling the precision–recall trade-off. Results should always specify the sacrebleu configuration used.
- Because ChrF operates at the character level, it is less interpretable than word-level metrics: a given change in ChrF score does not have an obvious linguistic meaning.
- ChrF does not account for word order beyond what is captured incidentally by character $n$-grams (unless ChrF++ with ``word_order > 0`` is used).
- Like all reference-based metrics, ChrF penalises valid paraphrases that diverge from the reference at the character level.