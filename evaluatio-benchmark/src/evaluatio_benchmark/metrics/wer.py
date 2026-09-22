from dataclasses import dataclass
from typing import List

import datasets
import evaluate
import evaluatio.metrics.wer
import jiwer
import numpy as np
import pandas as pd
from evaluatio_benchmark.benchmark import BenchResult, benchmark


@dataclass
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float


def _get_dataset() -> pd.DataFrame:
    ds: datasets.Dataset = datasets.load_dataset(
        "techiaith/evals-speech-recognition-cy-en",
        name="cymen_arfor__lleisiau_arfor",
        split="openai__whisper_large_v3__main",
    ).select_columns(["sentence", "prediction"])
    return ds.to_pandas().dropna()  # type: ignore


def wer_benchmark() -> List[BenchResult]:
    df = _get_dataset()
    references = df["sentence"].to_list()  # type: ignore
    hypotheses = df["prediction"].to_list()  # type: ignore

    ## Functions for WER benchmark
    def evaluatio_wer() -> float:
        return evaluatio.metrics.wer.word_error_rate(references, hypotheses)

    def jiwer_wer() -> float:
        return jiwer.wer(references, hypotheses)

    wer = evaluate.load("wer")

    def evaluate_wer() -> float:
        return wer.compute(predictions=hypotheses, references=references)  # type: ignore

    return [benchmark(evaluatio_wer), benchmark(jiwer_wer), benchmark(evaluate_wer)]


def wer_per_pair_benchmark() -> List[BenchResult]:
    # Limit to 200 since evaluate is too slow to handle much more
    df = _get_dataset()  #  .head(200)
    references = df["sentence"].to_list()  # type: ignore
    hypotheses = df["prediction"].to_list()  # type: ignore

    ## Functions for WER benchmark
    def evaluatio_wer_per_pair() -> List[float]:
        return evaluatio.metrics.wer.word_error_rate_per_pair(references, hypotheses)

    def jiwer_wer_per_pair() -> List[float]:
        return [jiwer.wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]

    wer = evaluate.load("wer")

    def evaluate_wer_per_pair() -> List[float]:
        return [
            wer.compute(predictions=[pred], references=[ref])
            for ref, pred in zip(references, hypotheses)
        ]  # type: ignore

    return [
        benchmark(evaluatio_wer_per_pair),
        benchmark(jiwer_wer_per_pair),
        benchmark(evaluate_wer_per_pair, repeats=2, warmup=1),
    ]


def wer_ci_benchmark() -> List[BenchResult]:
    # Limit to 200 since evaluate is too slow to handle much more
    df = _get_dataset()  # .head(200)
    references = df["sentence"].to_list()  # type: ignore
    hypotheses = df["prediction"].to_list()  # type: ignore

    ITERATIONS = 5_000
    ALPHA = 0.05

    def evaluatio_word_error_rate_ci() -> "ConfidenceInterval":
        return evaluatio.metrics.wer.word_error_rate_ci(
            references, hypotheses, ITERATIONS, ALPHA
        )  # type: ignore

    def jiwer_word_error_rate_ci():
        assert len(references) == len(hypotheses)
        assert len(references) > 0
        assert 0 < ALPHA < 1

        n = len(references)

        # Compute per-pair edit distance and ref lengths once
        edit_distances = []
        ref_lengths = []

        for ref, hyp in zip(references, hypotheses):
            output = jiwer.process_words(ref, hyp)
            distance = output.substitutions + output.insertions + output.deletions
            edit_distances.append(distance)
            ref_lengths.append(len(ref.split()))

        edit_distances = np.array(edit_distances)
        ref_lengths = np.array(ref_lengths)

        # Observed corpus-level WER
        total_edits = edit_distances.sum()
        total_refs = ref_lengths.sum()
        mean = total_edits / total_refs

        # Bootstrap
        bootstrapped = np.empty(ITERATIONS)

        for i in range(ITERATIONS):
            idx = np.random.randint(0, n, size=n)
            sum_edits = edit_distances[idx].sum()
            sum_refs = ref_lengths[idx].sum()
            bootstrapped[i] = sum_edits / sum_refs

        bootstrapped.sort()

        lower_idx = int(np.floor((ALPHA / 2) * ITERATIONS))
        upper_idx = int(np.floor((1 - ALPHA / 2) * ITERATIONS))

        lower = bootstrapped[min(lower_idx, ITERATIONS - 1)]
        upper = bootstrapped[min(upper_idx, ITERATIONS - 1)]

        return ConfidenceInterval(mean, lower, upper)

    return [
        benchmark(evaluatio_word_error_rate_ci),
        benchmark(jiwer_word_error_rate_ci),
    ]
