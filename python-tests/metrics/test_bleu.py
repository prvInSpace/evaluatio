import pytest
from evaluatio.metrics.bleu import bleu_bootstrap_test


@pytest.fixture
def identical_references_and_hypotheses():
    references = [["the cat sat on the mat"], ["the dog ate the bone"]]
    hyp = ["the cat sat on the mat", "the dog ate the bone"]
    return references, hyp


@pytest.fixture
def clearly_different_hypotheses():
    references = [["the cat sat on the mat"]] * 50
    hyp1 = ["the cat sat on the mat"] * 50  # perfect
    hyp2 = ["a completely wrong translation"] * 50
    return references, hyp1, hyp2


def test_empty_references_raises():
    with pytest.raises(ValueError, match="empty"):
        bleu_bootstrap_test([], [], [], iterations=999)


def test_mismatched_lengths_hyp1_raises():
    references = [["the cat sat on the mat"], ["the dog ate the bone"]]
    hyp1 = ["the cat sat on the mat"]
    hyp2 = ["the cat sat on the mat", "the dog ate the bone"]
    with pytest.raises(ValueError, match="same length"):
        bleu_bootstrap_test(references, hyp1, hyp2, iterations=999)


def test_mismatched_lengths_hyp2_raises():
    references = [["the cat sat on the mat"], ["the dog ate the bone"]]
    hyp1 = ["the cat sat on the mat", "the dog ate the bone"]
    hyp2 = ["the cat sat on the mat"]
    with pytest.raises(ValueError, match="same length"):
        bleu_bootstrap_test(references, hyp1, hyp2, iterations=999)


def test_generator_references_accepted(clearly_different_hypotheses):
    # Generators must be materialised without error — the key regression
    # being tested here is that references is not consumed on first zip
    references, hyp1, hyp2 = clearly_different_hypotheses
    result = bleu_bootstrap_test((r for r in references), hyp1, hyp2, iterations=999)
    assert isinstance(result, float)


def test_generator_hypotheses_accepted(clearly_different_hypotheses):
    references, hyp1, hyp2 = clearly_different_hypotheses
    result = bleu_bootstrap_test(
        references,
        (h for h in hyp1),
        (h for h in hyp2),
        iterations=999,
    )
    assert isinstance(result, float)


def test_multiple_references_per_sentence_accepted():
    references = [
        ["the cat sat on the mat", "a cat sat on the mat"],
        ["the dog ate the bone", "a dog ate the bone"],
    ]
    hyp1 = ["the cat sat on the mat", "the dog ate the bone"]
    hyp2 = ["a cat sat on a mat", "a dog ate a bone"]
    result = bleu_bootstrap_test(references, hyp1, hyp2, iterations=999)
    assert isinstance(result, float)


def test_returns_float(clearly_different_hypotheses):
    references, hyp1, hyp2 = clearly_different_hypotheses
    result = bleu_bootstrap_test(references, hyp1, hyp2, iterations=999)
    assert isinstance(result, float)


def test_p_value_in_valid_range(clearly_different_hypotheses):
    references, hyp1, hyp2 = clearly_different_hypotheses
    result = bleu_bootstrap_test(references, hyp1, hyp2, iterations=999)
    assert 0.0 < result <= 1.0


def test_minimum_p_value_floor(clearly_different_hypotheses):
    # With iterations=999, minimum possible p is 1/1000
    references, hyp1, hyp2 = clearly_different_hypotheses
    result = bleu_bootstrap_test(references, hyp1, hyp2, iterations=999)
    assert result >= 0.001


def test_identical_systems_not_significant(identical_references_and_hypotheses):
    references, hyp = identical_references_and_hypotheses
    result = bleu_bootstrap_test(references, hyp, hyp, iterations=4999)
    assert result > 0.5, f"Expected p > 0.5 for identical systems, got {result}"


def test_clearly_different_systems_significant(clearly_different_hypotheses):
    references, hyp1, hyp2 = clearly_different_hypotheses
    result = bleu_bootstrap_test(references, hyp1, hyp2, iterations=4999)
    assert result < 0.05, (
        f"Expected p < 0.05 for clearly different systems, got {result}"
    )


def test_symmetry(clearly_different_hypotheses):
    # Swapping hyp1 and hyp2 should give the same p-value since the
    # function orients better vs worse internally
    references, hyp1, hyp2 = clearly_different_hypotheses
    p_12 = bleu_bootstrap_test(references, hyp1, hyp2, iterations=9999)
    p_21 = bleu_bootstrap_test(references, hyp2, hyp1, iterations=9999)
    assert abs(p_12 - p_21) < 0.05, f"Expected symmetric p-values, got {p_12} vs {p_21}"
