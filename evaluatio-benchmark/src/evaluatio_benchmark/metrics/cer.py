from dataclasses import dataclass
from typing import List

import datasets
import evaluate
import evaluatio.metrics.cer
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


def cer_benchmark() -> List[BenchResult]:
    df = _get_dataset()
    references = df["sentence"].to_list()  # type: ignore
    hypotheses = df["prediction"].to_list()  # type: ignore

    ## Functions for cer benchmark
    def evaluatio_cer() -> float:
        return evaluatio.metrics.cer.character_error_rate(references, hypotheses)

    def jiwer_cer() -> float:
        return jiwer.cer(references, hypotheses)

    cer = evaluate.load("cer")

    def evaluate_cer() -> float:
        return cer.compute(predictions=hypotheses, references=references)  # type: ignore

    return [benchmark(evaluatio_cer), benchmark(jiwer_cer), benchmark(evaluate_cer)]


def cer_per_pair_benchmark() -> List[BenchResult]:
    # Limit to 200 since evaluate is too slow to handle much more
    df = _get_dataset()  # .head(200)
    references = df["sentence"].to_list()  # type: ignore
    hypotheses = df["prediction"].to_list()  # type: ignore

    ## Functions for cer benchmark
    def evaluatio_cer_per_pair() -> List[float]:
        return evaluatio.metrics.cer.character_error_rate_per_pair(
            references, hypotheses
        )

    def jiwer_cer_per_pair() -> List[float]:
        return [jiwer.cer(ref, hyp) for ref, hyp in zip(references, hypotheses)]

    cer = evaluate.load("cer")

    def evaluate_cer_per_pair() -> List[float]:
        return [
            cer.compute(predictions=[pred], references=[ref])
            for ref, pred in zip(references, hypotheses)
        ]  # type: ignore

    return [
        benchmark(evaluatio_cer_per_pair),
        benchmark(jiwer_cer_per_pair),
        benchmark(evaluate_cer_per_pair, repeats=2, warmup=1),
    ]


def cer_ci_benchmark() -> List[BenchResult]:
    # Limit to 200 since evaluate is too slow to handle much more
    df = _get_dataset()  # .head(200)
    references = df["sentence"].to_list()  # type: ignore
    hypotheses = df["prediction"].to_list()  # type: ignore

    ITERATIONS = 5_000
    ALPHA = 0.05

    def evaluatio_character_error_rate_ci() -> "ConfidenceInterval":
        return evaluatio.metrics.cer.character_error_rate_ci(
            references, hypotheses, ITERATIONS, ALPHA
        )  # type: ignore

    def jiwer_character_error_rate_ci():
        assert len(references) == len(hypotheses)
        assert len(references) > 0
        assert 0 < ALPHA < 1

        n = len(references)

        # Compute per-pair edit distance and ref lengths once
        edit_distances = []
        ref_lengths = []

        for ref, hyp in zip(references, hypotheses):
            output = jiwer.process_characters(ref, hyp)
            distance = output.substitutions + output.insertions + output.deletions
            edit_distances.append(distance)
            ref_lengths.append(len(ref.split()))

        edit_distances = np.array(edit_distances)
        ref_lengths = np.array(ref_lengths)

        # Observed corpus-level cer
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

        locer_idx = int(np.floor((ALPHA / 2) * ITERATIONS))
        upper_idx = int(np.floor((1 - ALPHA / 2) * ITERATIONS))

        locer = bootstrapped[min(locer_idx, ITERATIONS - 1)]
        upper = bootstrapped[min(upper_idx, ITERATIONS - 1)]

        return ConfidenceInterval(mean, locer, upper)

    return [
        benchmark(evaluatio_character_error_rate_ci),
        benchmark(jiwer_character_error_rate_ci),
    ]
