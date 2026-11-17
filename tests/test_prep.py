"""Binary event selection and validation of prepared model outputs."""

from copy import deepcopy

import numpy as np
import pytest

from pysmile.prep import USbind_out, USprep_mdl


class ProbabilityModel:
    def __init__(self, classes=(0, 1), probabilities=((.8, .2), (.1, .9))):
        self.classes_ = np.array(classes)
        self.probabilities = np.array(probabilities)

    def predict_proba(self, X):
        return self.probabilities


@pytest.mark.parametrize("classes", [(1, 0), ("1", "0"), (True, False)])
def test_preparation_selects_event_by_class_label(classes):
    model = ProbabilityModel(classes, ((.2, .8), (.9, .1)))
    result = USprep_mdl(model, np.zeros((2, 1)), [0, 1])
    np.testing.assert_array_equal(result["p"], [.2, .9])


@pytest.mark.parametrize("classes, probabilities", [
    ((0,), ((1,), (1,))), ((0, 1, 2), ((.2, .3, .5), (.2, .3, .5))),
    ((1, 2), ((.8, .2), (.1, .9))), ((0, 0), ((.8, .2), (.1, .9))),
    ((0, 1), ((1,), (1,))), ((0, 1), (.2, .9)),
    ((0, 1), ((.8, np.nan), (.1, .9))), ((0, 1), ((.8, 1.5), (.1, .9))),
])
def test_preparation_rejects_unsupported_probability_outputs(classes, probabilities):
    with pytest.raises(ValueError):
        USprep_mdl(ProbabilityModel(classes, probabilities), np.zeros((2, 1)), [0, 1])


def test_preparation_requires_explicit_model_classes():
    model = ProbabilityModel()
    del model.classes_
    with pytest.raises(ValueError, match="classes_"):
        USprep_mdl(model, np.zeros((2, 1)), [0, 1])


@pytest.mark.parametrize("y", [[0], [0, 0], [0, 2], [0, np.nan], [[0, 1]]])
def test_preparation_rejects_invalid_or_misaligned_outcomes(y):
    with pytest.raises(ValueError):
        USprep_mdl(ProbabilityModel(), np.zeros((2, 1)), y)


def test_preparation_preserves_optional_outcomes_and_rounding_tolerance():
    model = ProbabilityModel(probabilities=((1, -1e-10), (0, 1 + 1e-10)))
    result = USprep_mdl(model, np.zeros((2, 1)))
    assert result["y"] is None
    np.testing.assert_array_equal(result["p"], [0, 1])


@pytest.mark.parametrize("field, value", [
    ("y", None), ("y", []), ("y", [0, 2]), ("y", [0, np.nan]),
    ("p", [.2]), ("p", [.2, 1.1]), ("p", [.2, np.nan]),
    ("n_vars", -1), ("n_vars", np.nan), ("n_vars", np.inf), ("n_vars", True),
])
def test_binding_rejects_invalid_prepared_outputs(field, value):
    ref = {"y": [0, 1], "p": [.2, .9], "n_vars": 1}
    with pytest.raises(ValueError):
        USbind_out(ref, {**ref, field: value})


def test_binding_normalizes_binary_encodings_without_mutation():
    ref = {"y": ["0", "1"], "p": [.2, .9], "n_vars": np.uint64(1)}
    new = {"y": [0, 1], "p": [.1, .8], "n_vars": np.uint64(2)}
    before = deepcopy((ref, new))
    result = USbind_out(ref, new)
    assert result["n_vars_diff"] == 1
    np.testing.assert_array_equal(result["comparison_df"].y, [0, 1])
    assert (ref, new) == before


@pytest.mark.parametrize("output", [None, {}, {"y": [0, 1], "p": [.2, .9]}])
def test_binding_requires_prepared_output_fields(output):
    with pytest.raises(ValueError):
        USbind_out(output, output)


def test_named_dataset_preparation_and_variable_metadata():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    frame = pd.DataFrame({"age": [-2., -1., 0., 1.], "event": [0, 1, 0, 1]})
    model = LogisticRegression().fit(frame[["age"]], frame.event)
    actual = USprep_mdl(model, dataset=frame, testing=True, target_name="event")
    assert actual["var_names"] == ["age"]
    np.testing.assert_array_equal(actual["y"], frame.event)
    bound = USbind_out(actual, actual)
    assert bound["ref_vars"] == bound["new_vars"] == ["age"]
    assert "event" in frame


def test_pipeline_counts_transformed_features():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LogisticRegression
    X = np.array([[-2.], [-1.], [0.], [1.]])
    y = [0, 1, 0, 1]
    model = make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression()).fit(X, y)
    actual = USprep_mdl(model, X, y)
    assert actual["n_vars"] == 2
    assert actual["var_names"] == ["x0", "x0^2"]
