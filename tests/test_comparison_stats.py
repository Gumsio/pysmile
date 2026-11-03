"""R reference and mathematical checks for ROC and calibration summaries."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pysmile.stats import _roc_statistics, _calibration_statistics


REFERENCE = Path(__file__).parent / "reference"


@pytest.fixture
def prepared():
    data = pd.read_csv(REFERENCE / "predictions.tsv", sep="\t")
    return ({"y": data.y.to_numpy(), "p": data.p_ref.to_numpy(), "n_vars": 1},
            {"y": data.y.to_numpy(), "p": data.p.to_numpy(), "n_vars": 2})


@pytest.mark.parametrize("name, calculate", [("roc", _roc_statistics),
                                            ("calibration", _calibration_statistics)])
def test_comparison_matches_independent_r_curves_and_summaries(prepared, name, calculate):
    before = deepcopy(prepared)
    curves, results = calculate(*prepared, **({"n_bins": 5} if name == "calibration" else {}))
    summary = pd.read_csv(REFERENCE / f"{name}_results.tsv", sep="\t", dtype=str)
    assert [f"{results[key]:.3f}" for key in summary.Metric] == list(summary.Value)
    reference = pd.read_csv(REFERENCE / f"{name}_curves.tsv", sep="\t").rename(columns={
        "FalsePositiveRate": "fpr", "Sensitivity": "tpr",
    })
    for actual, model in zip(curves, ["Reference", "New"]):
        expected = reference.loc[reference.Model == model]
        if name == "calibration":
            expected = expected.sort_values("Predicted", kind="stable")
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-14)
    for actual, expected in zip(prepared, before):
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_delong_matches_direct_pairwise_comparison_with_ties(alternative):
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    probabilities = ([.1, .8, .5, .5, .4, .4, .9, .9], [.2, .7, .3, .6, .8, .4, .3, .6])
    outputs = [{"y": y, "p": p, "n_vars": 1} for p in probabilities]
    _, results = _roc_statistics(*outputs, alternative=alternative)
    placements = []
    aucs = []
    for p in probabilities:
        p = np.array(p)
        differences = p[y == 1, None] - p[None, y == 0]
        kernel = (differences > 0).astype(float) + .5 * (differences == 0)
        placements.append((kernel.mean(axis=1), kernel.mean(axis=0)))
        aucs.append(kernel.mean())
    variance = (np.var(placements[0][0] - placements[1][0], ddof=1) / np.sum(y == 1) +
                np.var(placements[0][1] - placements[1][1], ddof=1) / np.sum(y == 0))
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    expected = {"two.sided": 2 * norm.sf(abs(z)), "greater": norm.sf(z), "less": norm.cdf(z)}
    np.testing.assert_allclose(results["AUC_ref"], aucs[0])
    np.testing.assert_allclose(results["AUC_new"], aucs[1])
    np.testing.assert_allclose(results["AUC_diff_p_value"], expected[alternative])


@pytest.mark.parametrize("probabilities, auc, individual_p", [([0, 0, 1, 1], 1., 0.),
                                                            ([.5] * 4, .5, 1.)])
def test_roc_zero_variance(probabilities, auc, individual_p):
    output = {"y": [0, 0, 1, 1], "p": probabilities, "n_vars": 0}
    _, results = _roc_statistics(output, output)
    assert results["AUC_ref"] == auc
    assert results["AUC_p-value_ref"] == individual_p
    assert results["AUC_diff_p_value"] == 1
    for alternative in ["less", "greater"]:
        assert _roc_statistics(output, output, alternative)[1]["AUC_diff_p_value"] == .5


def test_roc_single_observation_variance_is_missing():
    output = {"y": [0, 0, 1], "p": [.2, .5, .9], "n_vars": 0}
    _, results = _roc_statistics(output, output)
    assert results["AUC_ref"] == 1
    assert np.isnan(results["AUC_p-value_ref"])
    assert np.isnan(results["AUC_diff_p_value"])


@pytest.mark.parametrize("n_bins", [0, 1, 11, 2.5, None, True, np.nan, np.inf, [5]])
def test_calibration_rejects_invalid_bin_count(prepared, n_bins):
    with pytest.raises(ValueError):
        _calibration_statistics(*prepared, n_bins=n_bins)


def test_calibration_right_closed_bins_include_endpoints():
    output = {"y": [0, 0, 1, 1], "p": [0, .5, .5, 1], "n_vars": 1}
    curves, _ = _calibration_statistics(output, output, 2)
    np.testing.assert_array_equal(curves[0].Count, [3, 1])
    np.testing.assert_allclose(curves[0].Predicted, [1 / 3, 1])
    np.testing.assert_allclose(curves[0].Empirical, [1 / 3, 1])


def test_calibration_perfect_reference_has_missing_skill_score():
    ref = {"y": [0, 0, 1, 1], "p": [0, 0, 1, 1], "n_vars": 1}
    new = {**ref, "p": [.2, .3, .7, .8]}
    _, results = _calibration_statistics(ref, new, 2)
    assert results["Brier_ref"] == 0
    assert results["Delta_Brier"] < 0
    assert np.isnan(results["BSS"])


@pytest.mark.parametrize("calculate", [_roc_statistics, _calibration_statistics])
def test_comparison_rejects_misaligned_outcomes(prepared, calculate):
    ref, new = prepared
    with pytest.raises(ValueError, match="Target values differ"):
        calculate(ref, {**new, "y": 1 - new["y"]})
