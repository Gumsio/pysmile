import numpy as np
import pandas as pd

# prepare the fitted model for usmile
# by extracting observed target values and probabilities
#
# should work with sklearn
# should work for binary target class
#
# functionality of R's function of the same name
def USprep_mdl(model, X, y=None, target_name=None):
    
    # extract probabilities of model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        p = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    
    else:
        raise ValueError("Invalid model! Does not have predict_proba() method!")


    # extract dof of model
    if hasattr(model, "coef_"):
        n_vars = model.coef_.shape[1]
    elif hasattr(model, "feature_importances_"):
        n_vars = len(model.feature_importances_)
    elif hasattr(X, "shape"):
        n_vars = X.shape[1]
    else:
        raise ValueError("Can't read dof of fitted model!")


    return {
        "y": np.array(y) if y is not None else None,
        "p": np.clip(p, 0.0, 1.0),
        "n_vars": n_vars, 
    }


# combine outputs of reference and new models together for usmile
#
# functionality of R's function of the same name
def USbind_out(model_out_ref, model_out_new):
    
    y_ref = model_out_ref["y"]
    y_new = model_out_new["y"]

    if y_ref is None or y_new is None: raise ValueError("Invalid models! One or both models have no target values!")
    if not np.array_equal(y_ref, y_new): raise ValueError("Target values differ between reference and new model!")

    comparison_df = pd.DataFrame({
        "y": y_ref,
        "p_ref": model_out_ref["p"],
        "p": model_out_new["p"],
    })

    n_vars_diff = abs(model_out_ref["n_vars"] - model_out_new["n_vars"])

    return {
        "comparison_df": comparison_df,
        "n_vars_diff": n_vars_diff,
    }