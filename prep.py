import numpy as np
import pandas as pd
from collections.abc import Mapping


def _binary_outcomes(y, both_classes=False):
    y = np.asarray(y)
    if y.ndim != 1 or not y.size:
        raise ValueError("Target values must be a nonempty binary vector!")
    if y.dtype.kind in "OUS":
        if not all(isinstance(value, str) and value in ["0", "1"] for value in y):
            raise ValueError("Target values must be coded as 0 and 1!")
        y = y.astype(int)
    if y.dtype.kind not in "biuf" or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Target values must be coded as 0 and 1!")
    if both_classes and np.unique(y).size != 2:
        raise ValueError("Target values must contain both outcome classes!")
    return y


def _probabilities(p, allow_empty=False, tolerance=0.0):
    p = np.asarray(p)
    if p.ndim != 1 or p.dtype.kind not in "biuf" or (not allow_empty and not p.size):
        raise ValueError("Probabilities must be a numeric vector!")
    if not np.isfinite(p).all() or np.any(p < -tolerance) or np.any(p > 1 + tolerance):
        raise ValueError("Probabilities must be finite and between 0 and 1!")
    return p.astype(float, copy=False)


def _variable_count(value, name):
    value = np.asarray(value)
    if value.ndim != 0 or value.dtype.kind not in "iuf" or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be one finite, non-negative number!")
    return value.item()

# prepare the fitted model for usmile
# by extracting observed target values and probabilities
#
# should work with sklearn
# should work for binary target class
#
# functionality of R's function of the same name
def USprep_mdl(model, X=None, y=None, target_name=None, *, calibrator=None, dataset=None, testing=False):

    if not isinstance(testing, (bool, np.bool_)):
        raise ValueError("testing must be True or False!")
    if dataset is not None:
        if X is not None:
            raise ValueError("Supply X or dataset, not both!")
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("dataset must be a DataFrame!")
        X = dataset
    if X is None:
        raise ValueError("Supply evaluation data; this model does not expose stored training data!")

    if target_name is not None:
        if not isinstance(target_name, str) or not isinstance(X, pd.DataFrame) or target_name not in X.columns:
            raise ValueError("target_name must identify an outcome column in the input DataFrame!")
        target = _binary_outcomes(X[target_name], both_classes=True)
        if y is not None and not np.array_equal(_binary_outcomes(y, both_classes=True), target):
            raise ValueError("Supplied target values differ from target_name!")
        y = target
        X = X.drop(columns=target_name)
    
    # extract probabilities of model
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X))
        classes = np.asarray(getattr(model, "classes_", None))
        if classes.ndim != 1 or classes.size != 2:
            raise ValueError("Model must identify two outcome classes through classes_!")
        classes = _binary_outcomes(classes, both_classes=True)
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError("Model must return two columns of binary class probabilities!")
        p = probs[:, np.flatnonzero(classes == 1)[0]]
        p = _probabilities(p, tolerance=np.sqrt(np.finfo(float).eps))
    
    else:
        raise ValueError("Invalid model! Does not have predict_proba() method!")


    # extract dof of model
    fitted_model = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(fitted_model, "coef_"):
        n_vars = fitted_model.coef_.shape[1]
    elif hasattr(fitted_model, "feature_importances_"):
        n_vars = len(fitted_model.feature_importances_)
    elif hasattr(fitted_model, "n_features_in_"):
        n_vars = fitted_model.n_features_in_
    elif hasattr(X, "shape"):
        n_vars = X.shape[1]
    else:
        raise ValueError("Can't read dof of fitted model!")

    n_vars = _variable_count(n_vars, "n_vars")
    if y is not None:
        y = _binary_outcomes(y, both_classes=True)
        if len(y) != len(p):
            raise ValueError("Target values and probabilities must have the same length!")

    var_names = list(getattr(model, "feature_names_in_", []))
    if hasattr(model, "steps") and len(model.steps) > 1:
        try:
            var_names = list(model[:-1].get_feature_names_out())
        except (AttributeError, ValueError):
            var_names = []
    elif not var_names and isinstance(X, pd.DataFrame):
        var_names = list(X.columns)
    if len(var_names) != n_vars:
        var_names = []

    if calibrator is not None:
        if __package__:
            from .calib import USapply_calibrator
        else:
            from calib import USapply_calibrator
        if getattr(calibrator, "event_level", None) != "1":
            raise ValueError("Calibrator must represent probabilities for event class 1!")
        p = USapply_calibrator(calibrator, np.clip(p, 0.0, 1.0))

    return {
        "y": y,
        "p": np.clip(p, 0.0, 1.0),
        "n_vars": n_vars, 
        "var_names": var_names,
    }


# combine outputs of reference and new models together for usmile
#
# functionality of R's function of the same name
def USbind_out(model_out_ref, model_out_new):

    for model_out in [model_out_ref, model_out_new]:
        if not isinstance(model_out, Mapping) or not {"y", "p", "n_vars"}.issubset(model_out):
            raise ValueError("Model outputs must contain y, p and n_vars!")
        if model_out["y"] is None:
            raise ValueError("Invalid models! One or both models have no target values!")
        outcomes = _binary_outcomes(model_out["y"])
        probabilities = _probabilities(model_out["p"])
        if len(outcomes) != len(probabilities):
            raise ValueError("Target values and probabilities must have the same length!")
        _variable_count(model_out["n_vars"], "n_vars")
    
    y_ref = _binary_outcomes(model_out_ref["y"])
    y_new = _binary_outcomes(model_out_new["y"])

    if not np.array_equal(y_ref, y_new): raise ValueError("Target values differ between reference and new model!")

    comparison_df = pd.DataFrame({
        "y": y_ref,
        "p_ref": model_out_ref["p"],
        "p": model_out_new["p"],
    })

    n_vars_diff = abs(_variable_count(model_out_ref["n_vars"], "n_vars") -
                      _variable_count(model_out_new["n_vars"], "n_vars"))

    return {
        "comparison_df": comparison_df,
        "n_vars_diff": n_vars_diff,
        "ref_vars": list(model_out_ref.get("var_names", [])),
        "new_vars": list(model_out_new.get("var_names", [])),
    }
