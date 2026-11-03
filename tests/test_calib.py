"""Calibration input checks and fitted probability behavior."""

import numpy as np
import pytest
import pandas as pd
from pathlib import Path

from pysmile.calib import USCalibrator
from pysmile.calib import USfit_calibrator, USapply_calibrator


METHODS = ("logistic", "intercept", "isotonic")
OUTCOMES = [0, 0, 1, 0, 1, 1]
PROBABILITIES = [.1, .25, .4, .6, .75, .9]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("p", [
    [-.1, .25, .4, .6, .75, .9], [.1, .25, .4, .6, .75, 1.1],
    [.1, .25, np.nan, .6, .75, .9], [.1, .25, np.inf, .6, .75, .9],
    [], [.2, .8], [[.1, .25, .4, .6, .75, .9]], [".1"] * 6,
])
def test_calibrator_rejects_invalid_fit_probabilities(method, p):
    with pytest.raises(ValueError):
        USCalibrator(method).fit(OUTCOMES, p)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("y", [[0] * 6, [1] * 6, [0, 0, 2, 0, 1, 1],
                                   [0, 0, np.nan, 0, 1, 1], [[0, 0, 1, 0, 1, 1]]])
def test_calibrator_rejects_invalid_outcomes(method, y):
    with pytest.raises(ValueError):
        USCalibrator(method).fit(y, PROBABILITIES)


@pytest.mark.parametrize("eps", [0, -.1, .5, 1, np.nan, np.inf, None, True, "0.1", [.1]])
def test_calibrator_rejects_invalid_probability_restriction(eps):
    with pytest.raises(ValueError):
        USCalibrator(eps=eps)


@pytest.mark.parametrize("method", METHODS)
def test_calibrator_requires_three_observations(method):
    with pytest.raises(ValueError, match="three"):
        USCalibrator(method).fit([0, 1], [.2, .8])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("p", [[-.1], [1.1], [np.nan], [np.inf], [[.5]], ["0.5"]])
def test_calibrator_rejects_invalid_application_probabilities(method, p):
    calibrator = USCalibrator(method).fit(OUTCOMES, PROBABILITIES)
    with pytest.raises(ValueError):
        calibrator.apply(p)


@pytest.mark.parametrize("method", METHODS)
def test_calibrator_empty_application_and_fit_requirement(method):
    calibrator = USCalibrator(method)
    with pytest.raises(ValueError, match="fitted"):
        calibrator.apply([.5])
    calibrator.fit(OUTCOMES, PROBABILITIES)
    assert calibrator.apply([]).shape == (0,)


def test_intercept_calibration_uses_aligned_observations():
    with pytest.raises(ValueError, match="same length"):
        USCalibrator("intercept").fit([0, 0, 0, 1], [.2, .8])
    p = np.array(PROBABILITIES)
    original = p.copy()
    calibrator = USCalibrator("intercept").fit(OUTCOMES, p)
    np.testing.assert_allclose(calibrator.apply(p).mean(), np.mean(OUTCOMES), atol=1e-12)
    np.testing.assert_array_equal(p, original)


@pytest.mark.parametrize("method", METHODS)
def test_calibrator_accepts_r_binary_encodings(method):
    y = np.array(OUTCOMES).astype(str)
    reference = USCalibrator(method).fit(OUTCOMES, PROBABILITIES).apply(PROBABILITIES)
    actual = USCalibrator(method).fit(y, PROBABILITIES).apply(PROBABILITIES)
    np.testing.assert_allclose(actual, reference)


def test_r_named_calibration_functions_and_sample_metadata():
    calibrator = USfit_calibrator(OUTCOMES, PROBABILITIES, method="intercept")
    assert calibrator.n == 6
    assert calibrator.n_events == calibrator.n_non_events == 3
    assert calibrator.event_level == "1"
    assert calibrator.prevalence == .5
    np.testing.assert_array_equal(calibrator.probability_range, [.1, .9])
    np.testing.assert_allclose(USapply_calibrator(calibrator, PROBABILITIES),
                               calibrator.apply(PROBABILITIES))
    assert "Calibration observations: 6" in repr(calibrator)
    with pytest.raises(ValueError):
        USapply_calibrator({}, PROBABILITIES)


def test_preparation_applies_independent_calibrator_without_refitting():
    from pysmile.prep import USprep_mdl
    from sklearn.linear_model import LogisticRegression
    X = np.arange(6.).reshape(-1, 1)
    model = LogisticRegression().fit(X, OUTCOMES)
    calibrator = USfit_calibrator(OUTCOMES, PROBABILITIES, method="intercept")
    before = calibrator.intercept_
    actual = USprep_mdl(model, X, OUTCOMES, calibrator=calibrator)
    expected = calibrator.apply(model.predict_proba(X)[:, 1])
    np.testing.assert_allclose(actual["p"], expected)
    assert calibrator.intercept_ == before


@pytest.mark.parametrize("case", ["ordinary", "ties", "endpoints"])
@pytest.mark.parametrize("method", ["logistic", "intercept"])
def test_calibrator_matches_independently_executed_r(case, method):
    reference = Path(__file__).parent / "reference"
    inputs = pd.read_csv(reference / "calibrator_inputs.tsv", sep="\t")
    outputs = pd.read_csv(reference / "calibrator_results.tsv", sep="\t")
    inputs = inputs.loc[inputs["case"] == case]
    expected = outputs.loc[(outputs["case"] == case) & (outputs.method == method)]
    calibrator = USfit_calibrator(inputs.y, inputs.p, method=method)
    np.testing.assert_allclose(calibrator.apply(expected.p), expected.calibrated, atol=1e-8, rtol=1e-7)


def test_logistic_calibration_rejects_unidentifiable_slope():
    with pytest.raises(ValueError, match="varying"):
        USfit_calibrator(OUTCOMES, [.5] * len(OUTCOMES))
