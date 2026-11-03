"""Numerical R reference checks and statistical input regression tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pysmile.stats import UScalc_mdl, UScalc_raw


REFERENCE = Path(__file__).parent / "reference"


@pytest.fixture
def predictions():
    return pd.read_csv(REFERENCE / "predictions.tsv", sep="\t")


@pytest.mark.parametrize("coefficient", ["rLR", "BA", "RB"])
def test_statistics_match_independent_r_values_and_summaries(predictions, coefficient):
    before = predictions.copy(deep=True)
    actual = UScalc_raw(predictions, coefficient, 1)
    values = pd.read_csv(REFERENCE / f"us_{coefficient}_values.tsv", sep="\t")
    summary = pd.read_csv(REFERENCE / f"us_{coefficient}_summary.tsv", sep="\t").iloc[0]
    np.testing.assert_allclose(actual["data_Y"].value, values.value, atol=1e-14)
    np.testing.assert_allclose(actual["point_sizes"], values.proportion, atol=1e-14)
    for key, expected in summary.items():
        np.testing.assert_allclose(actual[key], expected, atol=1e-14)
    pd.testing.assert_frame_equal(predictions, before)


@pytest.mark.parametrize("coefficient", ["rLR", "BA", "RB"])
@pytest.mark.parametrize("field, value", [
    ("y", 0), ("y", 2), ("y", np.nan),
    ("p", np.nan), ("p_ref", np.inf), ("p", -.1), ("p_ref", 1.1),
])
def test_statistics_reject_invalid_data(predictions, coefficient, field, value):
    data = predictions.assign(**{field: value})
    with pytest.raises(ValueError):
        UScalc_raw(data, coefficient, 1)


@pytest.mark.parametrize("diff", [-1, np.nan, np.inf, True, None, [1], "1"])
def test_statistics_reject_invalid_variable_difference(predictions, diff):
    with pytest.raises(ValueError):
        UScalc_raw(predictions, "rLR", diff)


@pytest.mark.parametrize("bootstrap", [
    [], {}, {"p_Y_nonev": .1}, {"p_Y_nonev": np.nan, "p_Y_event": .2},
    {"p_Y_nonev": -.1, "p_Y_event": .2}, {"p_Y_nonev": .1, "p_Y_event": 1.1},
    {"p_Y_nonev": [.1], "p_Y_event": .2}, {"p_Y_nonev": .1, "p_Y_event": True},
])
def test_statistics_reject_invalid_bootstrap_values(predictions, bootstrap):
    with pytest.raises(ValueError):
        UScalc_raw(predictions, "rLR", 1, bootstrap_p=bootstrap)


def test_statistics_apply_supplied_bootstrap_probabilities(predictions):
    actual = UScalc_raw(predictions, "rLR", 1, {"p_Y_nonev": .01, "p_Y_event": .2})
    assert actual["p_Y_nonev"] == .01
    assert actual["p_Y_event"] == .2
    assert actual["nonev_solid"] is True
    assert actual["event_solid"] is False


def test_statistics_require_nonempty_unique_data_columns(predictions):
    invalid = [None, {}, predictions.iloc[:0], predictions.drop(columns="y"),
               pd.concat([predictions, predictions[["p"]]], axis=1)]
    for data in invalid:
        with pytest.raises(ValueError):
            UScalc_raw(data, "BA", 1)


@pytest.mark.parametrize("encoding", [str, bool])
def test_statistics_accept_r_binary_encodings(predictions, encoding):
    actual = UScalc_raw(predictions.assign(y=predictions.y.astype(encoding)), "rLR", 1)
    expected = UScalc_raw(predictions, "rLR", 1)
    pd.testing.assert_frame_equal(actual["data_Y"], expected["data_Y"])
    assert actual["p_Y_overall"] == expected["p_Y_overall"]


def test_statistics_handle_unsigned_probability_endpoints():
    data = pd.DataFrame({"y": [0, 0, 1, 1], "p_ref": [1., 1., 0., 0.], "p": [0., 0., 1., 1.]})
    expected = UScalc_raw(data, "BA", 1)
    actual = UScalc_raw(data.astype({"p": np.uint8, "p_ref": np.uint8}), "BA", 1)
    pd.testing.assert_frame_equal(actual["data_Y"], expected["data_Y"])
    np.testing.assert_array_equal(actual["point_sizes"], expected["point_sizes"])


@pytest.mark.parametrize("coefficient", ["BA", "RB"])
def test_model_summary_includes_overall_p_value(coefficient):
    X = np.array([[-2, 0], [-1, 1], [0, -1], [.5, 1], [1, 0], [2, -1]])
    y = [0, 0, 1, 0, 1, 1]
    ref = LogisticRegression().fit(X[:, :1], y)
    new = LogisticRegression().fit(X, y)
    actual = UScalc_mdl(ref, new, coefficient, X[:, :1], X, y)
    p_value = actual["results"].set_index("label").loc["LR overall p-value", "coefficient"]
    assert np.isfinite(p_value)
    assert p_value == actual["plot_data"]["p_Y_overall"]
