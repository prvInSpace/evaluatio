# BLEURT

BLEURT (Bilingual Evaluation Understudy with Representations from Transformers) is a learned evaluation metric for natural language generation tasks, most commonly machine translation. Unlike surface-form metrics such as [ChrF](/metrics/chrf.md) or [BLEU](/metrics/bleu.md), BLEURT is based on a fine-tuned BERT model and produces scores that correlate more strongly with human judgements of translation quality.

BLEURT produces a scalar score per sentence pair. The corpus-level score is the mean of sentence-level scores, which makes it straightforward to use with Evaluatio's generic inference functions without any dedicated wrapper.

## Evaluatio implementation

BLEURT is not implemented directly in Evaluatio. To use BLEURT scores, install the [official BLEURT package](https://github.com/google-research/bleurt) and obtain a checkpoint. Sentence-level scores can then be computed and passed directly to Evaluatio's generic inference functions.

```python
from bleurt import score

checkpoint = "bleurt/test_checkpoint"
scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)
```

`scorer.score` returns a list of floats, one per sentence pair, which can be passed directly to the inference functions described below.

## Comparing BLEURT scores of different models

Because BLEURT produces a sentence-level score and the corpus-level score is the mean of these, Evaluatio's generic `paired_bootstrap_test` and `paired_permutation_test` functions can be used directly without any metric-specific wrapper.

### Confidence interval

As with model comparison, since the corpus-level BLEURT score is the mean of sentence-level scores, a bootstrap confidence interval on the corpus-level score can be estimated by resampling sentence-level scores and recomputing the mean. This is exactly what Evaluatio's generic `bootstrap_ci` function does.

### Example evaluation code

```python
import pandas as pd
from bleurt import score as bleurt_score

df = pd.read_csv("inferences.csv")

checkpoint = "bleurt/test_checkpoint"
scorer = bleurt_score.BleurtScorer(checkpoint)

model_1_scores = scorer.score(
    references=list(df["references"]),
    candidates=list(df["model_1"])
)
model_2_scores = scorer.score(
    references=list(df["references"]),
    candidates=list(df["model_2"])
)

from evaluatio.inference.hypothesis import paired_bootstrap_test
from evaluatio.inference.ci import bootstrap_ci

model_1_ci = bootstrap_ci(model_1_scores, iterations=5000, alpha=0.05)
model_2_ci = bootstrap_ci(model_2_scores, iterations=5000, alpha=0.05)

pvalue = paired_bootstrap_test(
    x1=model_1_scores,
    x2=model_2_scores,
    iterations=5000
)

def ci_plus_minus(ci):
    delta = ci.upper - ci.mean
    return f"{ci.mean:.3f} ± {delta:.3f}"

print(f"Model 1 BLEURT: {ci_plus_minus(model_1_ci)}")
print(f"Model 2 BLEURT: {ci_plus_minus(model_2_ci)}")
print(f"P-value: {pvalue}")
```

A paired permutation test can be used as an alternative to the bootstrap test:

```python
from evaluatio.inference.hypothesis import paired_permutation_test

pvalue = paired_permutation_test(
    x1=model_1_scores,
    x2=model_2_scores,
    iterations=5000
)
```

Both tests operate at the sentence level, preserving the paired dependency structure between model outputs.

### Why the generic functions are appropriate here

Unlike [WER](/metrics/wer), [ChrF](/metrics/chrf), or [BLEU](/metrics/bleu) where the corpus-level score is a ratio or aggregated count that cannot be recovered from the mean of sentence-level scores, BLEURT's corpus-level score is definitionally the mean of sentence-level scores. Bootstrapping over the mean is therefore exact rather than an approximation, and no metric-specific wrapper is needed.

### Multiple testing

When performing subgroup analyses (e.g. by domain, language pair, or document type), multiple statistical tests are often conducted simultaneously. Without correction, the probability of false positives increases.

See the page about [multiple testing](/inference/multiple_testing) for more info.

### Reporting recommendations

When reporting results for BLEURT comparisons, it is recommended to give the following:
- Mean BLEURT score for each model (and 95% CI)
- Paired bootstrap or permutation test $p$-value
- Absolute difference in mean BLEURT score

An example could look like:

> "Model A achieved a mean BLEURT score of 0.412 ± 0.009 (95% CI), compared to 0.374 ± 0.011 for Model B (paired bootstrap p=0.001), an absolute improvement of 0.038."

## Limitations of BLEURT

- BLEURT scores are not interpretable on an absolute scale and should only be compared within the same checkpoint and evaluation set.
- Results are sensitive to the choice of checkpoint; scores from different checkpoints are not directly comparable.
- BLEURT is computationally expensive relative to surface-form metrics, particularly on large evaluation sets.
- Like all reference-based metrics, BLEURT penalises valid paraphrases that diverge from the reference.