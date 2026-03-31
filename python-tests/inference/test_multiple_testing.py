from evaluatio.inference import multiple_testing
import pytest


def test_holm_alpha_out_of_bounds():
    with pytest.raises(ValueError):
        multiple_testing.holm_correction([0.4, 0.4], 0)
    with pytest.raises(ValueError):
        multiple_testing.holm_correction([0.4, 0.4], 1)


def test_holm_pvalue_out_of_bounds():
    multiple_testing.holm_correction([0.0, 1.0], 0.05)
    with pytest.raises(ValueError):
        multiple_testing.holm_correction([-0.1, 0.4], 1)
    with pytest.raises(ValueError):
        multiple_testing.holm_correction([1.1, 0.4], 1)


def test_bonferroni_alpha_out_of_bounds():
    with pytest.raises(ValueError):
        multiple_testing.bonferroni_correction([0.4, 0.4], 0)
    with pytest.raises(ValueError):
        multiple_testing.bonferroni_correction([0.4, 0.4], 1)


def test_bonferroni_pvalue_out_of_bounds():
    multiple_testing.bonferroni_correction([0.0, 1.0], 0.05)
    with pytest.raises(ValueError):
        multiple_testing.bonferroni_correction([-0.1, 0.4], 1)
    with pytest.raises(ValueError):
        multiple_testing.bonferroni_correction([1.1, 0.4], 1)
