# Getting Started

## Installation
Installing Evaluatio is quite straight forward:
=== "pip"
    ```bash
    pip install evaluatio
    ```
=== "uv"
    ```bash
    uv add evaluatio
    ```
=== "Poetry"
    ```bash
    poetry add evaluatio
    ```

If you want to use the translation metrics ([BLEU](/metrics/bleu), [ChrF](/metrics/chrf), etc.) you need to install [`sacrebleu`](https://pypi.org/project/sacrebleu/) as an optional dependency. This can be done by using the following commands:
=== "pip"
    ```bash
    pip install evaluatio[translation]
    ```
=== "uv"
    ```bash
    uv add evaluatio[translation]
    ```
=== "Poetry"
    ```bash
    poetry add evaluatio[translation]
    ```

## Example usage

The usage of the library depends quite heavily on what tasks you are doing and what systems you are evaluating.
The best place to start is therefore the tasks guides:

- [ASR Evaluation](/tasks/asr_evaluation)


### Calculating WER with confidence interval

Evaluatio is basically a drop-in replacement for [`evaluate`](https://pypi.org/project/evaluate/), especially for ASR related metrics:

```python
import pandas as pd
df = pd.read_csv("inferences.csv")

from evaluatio.metrics.wer import word_error_rate, word_error_rate_ci
model_wer = word_error_rate(df["references"], df["predictions"])
model_ci = word_error_rate_ci(df["references"], df["predictions"], 5000, 0.05)

print(f"Model WER: {model_wer}")
print(f"Model CI:  {model_ci.mean:.3f} ± {model_ci.upper-model_ci.mean:.3f}")
```