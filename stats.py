import numpy as np
import pandas as pd
import scipy.stats
from collections.abc import Mapping
if __package__:
    from .prep import USprep_mdl, USbind_out, _binary_outcomes, _probabilities, _variable_count
else:
    from prep import USprep_mdl, USbind_out, _binary_outcomes, _probabilities, _variable_count


# calculate statistics for Usmile from raw data
#
# functionality of R's function of the same name
def UScalc_raw(raw_data, y_coef, n_vars_diff, bootstrap_p=None):

    if not isinstance(y_coef, str) or y_coef not in ["rLR", "BA", "RB"]:
        raise ValueError("y_coef must be set to rLR, BA or RB!")

    if (not isinstance(raw_data, pd.DataFrame) or raw_data.columns.duplicated().any() or
            not {"y", "p_ref", "p"}.issubset(raw_data.columns)):
        raise ValueError("raw_data must be a DataFrame containing unique y, p_ref and p columns!")
    y = _binary_outcomes(raw_data["y"], both_classes=True)
    p_ref = _probabilities(raw_data["p_ref"])
    p = _probabilities(raw_data["p"])
    if len(y) != len(p_ref) or len(y) != len(p):
        raise ValueError("Target values and probabilities must have the same length!")
    n_vars_diff = _variable_count(n_vars_diff, "n_vars_diff")
    if bootstrap_p is not None:
        if not isinstance(bootstrap_p, Mapping) or not {"p_Y_nonev", "p_Y_event"}.issubset(bootstrap_p):
            raise ValueError("bootstrap_p must contain p_Y_nonev and p_Y_event!")
        for key in ["p_Y_nonev", "p_Y_event"]:
            value = np.asarray(bootstrap_p[key])
            if (value.ndim != 0 or value.dtype.kind not in "iuf" or
                    not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Bootstrap p-values must be finite probabilities between 0 and 1!")

    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    n = n0 + n1

    delta = p - p_ref

    flag = np.where(delta > 0, "up", np.where(delta < 0, "down", "c"))
    subclass = np.full(n, "unknown", dtype=object)

    # prediction subclasses based on probability shift
    subclass[(y == 0) & (flag == "down")] = "nonev_be"
    subclass[(y == 0) & (flag == "up")] = "nonev_wo"
    subclass[(y == 1) & (flag == "down")] = "event_wo"
    subclass[(y == 1) & (flag == "up")] = "event_be"

    I_nonev_be = np.sum(subclass == "nonev_be") / n0 if n0 > 0 else 0
    I_nonev_wo = np.sum(subclass == "nonev_wo") / n0 if n0 > 0 else 0
    I_event_wo = np.sum(subclass == "event_wo") / n1 if n1 > 0 else 0
    I_event_be = np.sum(subclass == "event_be") / n1 if n1 > 0 else 0

    netto_I_nonev = I_nonev_be - I_nonev_wo
    netto_I_event = I_event_be - I_event_wo
    I_overall = netto_I_nonev + netto_I_event

    df = max(n_vars_diff, 1)
    offset = 1e-10

    max_netto_Y_nonev, max_netto_Y_event, max_overall_Y, max_netto_Y_ratio = [np.nan, np.nan, np.nan, np.nan]
    p_Y_nonev, p_Y_event = [np.nan, np.nan]

    # R reports the overall likelihood-test p-value for every coefficient.
    res_ref = y * np.log(p_ref + offset) + (1 - y) * np.log(1 - p_ref + offset)
    res = y * np.log(p + offset) + (1 - y) * np.log(1 - p + offset)
    lrt_stat = 2 * abs(np.sum(res) - np.sum(res_ref))
    p_Y_overall = scipy.stats.chi2.sf(lrt_stat, df)

    # relative likelihood Ratio
    if y_coef == "rLR":
        SS_nonev_ref = np.sum(res_ref[y == 0])
        SS_event_ref = np.sum(res_ref[y == 1])
        SS_nonev = np.sum(res[y == 0])
        SS_event = np.sum(res[y == 1])
        
        SS_nonev_dw_ref = np.sum(res_ref[subclass == "nonev_be"])
        SS_nonev_up_ref = np.sum(res_ref[subclass == "nonev_wo"])
        SS_event_dw_ref = np.sum(res_ref[subclass == "event_wo"])
        SS_event_up_ref = np.sum(res_ref[subclass == "event_be"])
        
        SS_nonev_dw = np.sum(res[subclass == "nonev_be"])
        SS_nonev_up = np.sum(res[subclass == "nonev_wo"])
        SS_event_dw = np.sum(res[subclass == "event_wo"])
        SS_event_up = np.sum(res[subclass == "event_be"])
        
        nonev_be = -2 * (SS_nonev_dw_ref - SS_nonev_dw)
        nonev_wo = 2 * (SS_nonev_up_ref - SS_nonev_up)
        event_wo = 2 * (SS_event_dw_ref - SS_event_dw)
        event_be = -2 * (SS_event_up_ref - SS_event_up)
        
        netto_nonev = nonev_be - nonev_wo
        netto_event = event_be - event_wo
        overall = netto_nonev + netto_event
        
        max_netto_Y_nonev = -2 * SS_nonev_ref if nonev_be >= nonev_wo else -2 * SS_nonev
        max_netto_Y_event = -2 * SS_event_ref if event_be >= event_wo else -2 * SS_event
        max_overall_Y = -2 * (SS_nonev_dw_ref + SS_nonev_up_ref + SS_event_dw_ref + SS_event_up_ref)
        
        max_netto_Y_ratio = max_netto_Y_event / max_netto_Y_nonev if max_netto_Y_nonev != 0 else np.nan
        
        Y_nonev_be = nonev_be / max_netto_Y_nonev if max_netto_Y_nonev != 0 else 0
        Y_nonev_wo = nonev_wo / max_netto_Y_nonev if max_netto_Y_nonev != 0 else 0
        Y_event_wo = event_wo / max_netto_Y_event if max_netto_Y_event != 0 else 0
        Y_event_be = event_be / max_netto_Y_event if max_netto_Y_event != 0 else 0
        
        netto_Y_nonev = netto_nonev / max_netto_Y_nonev if max_netto_Y_nonev != 0 else 0
        netto_Y_event = netto_event / max_netto_Y_event if max_netto_Y_event != 0 else 0
        Y_overall = overall / max_overall_Y if max_overall_Y != 0 else 0
        
        if bootstrap_p is not None:
            p_Y_nonev = bootstrap_p["p_Y_nonev"]
            p_Y_event = bootstrap_p["p_Y_event"]
        else:
            shape = n_vars_diff / 2.0 if not np.isnan(n_vars_diff) else np.nan
            c1 = n1 / n
            c0 = 1 - c1
            rate0 = 1 / (2 * c0)
            rate1 = 1 / (2 * c1)
            
            if not np.isnan(shape):
                p_Y_event = scipy.stats.gamma.sf(abs(netto_event), a=shape, scale=1/rate1)
                p_Y_nonev = scipy.stats.gamma.sf(abs(netto_nonev), a=shape, scale=1/rate0)

    # Brier Alteration / Relative Brier
    elif y_coef in ["BA", "RB"]:
        res_sq_ref = (y - p_ref)**2
        res_sq = (p - y)**2
        
        SS_nonev_ref = np.sum(res_sq_ref[y == 0])
        SS_event_ref = np.sum(res_sq_ref[y == 1])
        
        delta_SS_nonev_dw = np.sum(res_sq_ref[subclass == "nonev_be"]) - np.sum(res_sq[subclass == "nonev_be"])
        delta_SS_nonev_up = np.sum(res_sq_ref[subclass == "nonev_wo"]) - np.sum(res_sq[subclass == "nonev_wo"])
        delta_SS_event_dw = np.sum(res_sq_ref[subclass == "event_wo"]) - np.sum(res_sq[subclass == "event_wo"])
        delta_SS_event_up = np.sum(res_sq_ref[subclass == "event_be"]) - np.sum(res_sq[subclass == "event_be"])
        
        if y_coef == "BA":
            Y_nonev_be = delta_SS_nonev_dw / n0 if n0 > 0 else 0
            Y_nonev_wo = -delta_SS_nonev_up / n0 if n0 > 0 else 0
            Y_event_wo = -delta_SS_event_dw / n1 if n1 > 0 else 0
            Y_event_be = delta_SS_event_up / n1 if n1 > 0 else 0
        else:
            if SS_nonev_ref == 0 or SS_event_ref == 0:
                raise ValueError("RB cannot be calculated: reference-model Brier component is zero.")
            Y_nonev_be = delta_SS_nonev_dw / SS_nonev_ref
            Y_nonev_wo = -delta_SS_nonev_up / SS_nonev_ref
            Y_event_wo = -delta_SS_event_dw / SS_event_ref
            Y_event_be = delta_SS_event_up / SS_event_ref
            
        netto_Y_nonev = Y_nonev_be - Y_nonev_wo
        netto_Y_event = Y_event_be - Y_event_wo
        Y_overall = ((n0 / n) * netto_Y_nonev) + ((n1 / n) * netto_Y_event)

    data_Y = pd.DataFrame({
        "subclass": ["nonev_be", "nonev_wo", "event_wo", "event_be"],
        "value": [Y_nonev_be, Y_nonev_wo, Y_event_wo, Y_event_be],
        "coefficient": ["Y"] * 4,
        "segment": ["nonev", "nonev", "event", "event"],
        "wo_be": [0, 1, 1, 0]
    })

    return {
        "data_Y": data_Y,
        "n": n, "n0": n0, "n1": n1,
        "point_sizes": [I_nonev_be, I_nonev_wo, I_event_wo, I_event_be],
        "netto_Y_nonev": netto_Y_nonev,
        "netto_Y_event": netto_Y_event,
        "Y_nonev_be": Y_nonev_be,
        "Y_nonev_wo": Y_nonev_wo,
        "Y_event_wo": Y_event_wo,
        "Y_event_be": Y_event_be,
        "nonev_solid": bool(not np.isnan(p_Y_nonev) and p_Y_nonev < 0.05) if y_coef == "rLR" else False,
        "event_solid": bool(not np.isnan(p_Y_event) and p_Y_event < 0.05) if y_coef == "rLR" else False,
        "Y_overall": Y_overall if y_coef == "rLR" else np.nan,
        "max_netto_Y_nonev": max_netto_Y_nonev,
        "max_netto_Y_event": max_netto_Y_event,
        "max_overall_Y": max_overall_Y,
        "max_netto_Y_ratio": max_netto_Y_ratio,
        "p_Y_nonev": p_Y_nonev,
        "p_Y_event": p_Y_event,
        "p_Y_overall": p_Y_overall,
        "df": df,
        "netto_I_nonev": netto_I_nonev,
        "netto_I_event": netto_I_event,
        "I_overall": I_overall
    }


# calculate Usmile statistics from two fitted sklearn models
#
# functionality of R's function of the same name
def UScalc_mdl(ref_model, new_model, y_coef, X_ref, X_new=None, y=None, ref_calibrator=None, new_calibrator=None):

    if X_new is None:
        X_new = X_ref
    
    ref_data = USprep_mdl(ref_model, X_ref, y, calibrator=ref_calibrator)
    new_data = USprep_mdl(new_model, X_new, y, calibrator=new_calibrator)

    combined = USbind_out(ref_data, new_data)
    plot_data = UScalc_raw(combined["comparison_df"], y_coef, combined["n_vars_diff"])

    results = pd.DataFrame({
        "label": [
            "df", "sample size", "number of non-events", "number of events",
            "non-events with better prediction", "non-events with worse prediction",
            "events with worse prediction", "events with better prediction",
            "netto non-events", "netto events", "overall",
            "max netto non-events", "max netto events", "max overall",
            "max events non-events ratio", "LR non-events p-value",
            "LR events p-value", "LR overall p-value",
            "I non-events with better prediction", "I non-events with worse prediction",
            "I events with worse prediction", "I events with better prediction",
            "I netto non-events", "I netto events", "I overall"
        ],
        "coefficient": [
            plot_data["df"], plot_data["n"], plot_data["n0"], plot_data["n1"],
            plot_data["Y_nonev_be"], plot_data["Y_nonev_wo"],
            plot_data["Y_event_wo"], plot_data["Y_event_be"],
            plot_data["netto_Y_nonev"], plot_data["netto_Y_event"],
            plot_data["Y_overall"] if y_coef == "rLR" else np.nan,
            plot_data["max_netto_Y_nonev"] if y_coef == "rLR" else np.nan,
            plot_data["max_netto_Y_event"] if y_coef == "rLR" else np.nan,
            plot_data["max_overall_Y"] if y_coef == "rLR" else np.nan,
            plot_data["max_netto_Y_ratio"] if y_coef == "rLR" else np.nan,
            plot_data["p_Y_nonev"] if y_coef == "rLR" else np.nan,
            plot_data["p_Y_event"] if y_coef == "rLR" else np.nan,
            plot_data["p_Y_overall"],
            plot_data["point_sizes"][0], plot_data["point_sizes"][1],
            plot_data["point_sizes"][2], plot_data["point_sizes"][3],
            plot_data["netto_I_nonev"], plot_data["netto_I_event"],
            plot_data["I_overall"]
        ]
    })

    return {
        "results": results,
        "plot_data": plot_data
    }


# calculate paired ROC statistics without fitting or changing either model
def _roc_statistics(data_ref, data_new, alternative="two.sided"):
    if not isinstance(alternative, str) or alternative not in ["two.sided", "greater", "less"]:
        raise ValueError("alternative must be two.sided, greater or less!")
    data = USbind_out(data_ref, data_new)["comparison_df"]
    y = _binary_outcomes(data["y"], both_classes=True)
    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)
    curves, positives, negatives, aucs = [], [], [], []

    for key in ["p_ref", "p"]:
        p = _probabilities(data[key])
        event = p[y == 1]
        nonevent = p[y == 0]
        ranks = scipy.stats.rankdata(np.r_[event, nonevent])
        v1 = (ranks[:n1] - scipy.stats.rankdata(event)) / n0
        v0 = 1 - (ranks[n1:] - scipy.stats.rankdata(nonevent)) / n1
        positives.append(v1)
        negatives.append(v0)
        aucs.append(np.mean(v1))

        # R orders thresholds from low to high, including both endpoints.
        unique, indices = np.unique(p, return_inverse=True)
        event_counts = np.bincount(indices[y == 1], minlength=len(unique))
        nonevent_counts = np.bincount(indices[y == 0], minlength=len(unique))
        curves.append(pd.DataFrame({
            "fpr": np.r_[1., (n0 - np.cumsum(nonevent_counts)) / n0],
            "tpr": np.r_[1., (n1 - np.cumsum(event_counts)) / n1],
        }))

    covariance = np.full((2, 2), np.nan)
    if n1 > 1 and n0 > 1:
        centered1 = np.asarray(positives) - np.asarray(aucs)[:, None]
        centered0 = np.asarray(negatives) - np.asarray(aucs)[:, None]
        covariance = (centered1 @ centered1.T / (n1 * (n1 - 1)) +
                      centered0 @ centered0.T / (n0 * (n0 - 1)))

    auc_p = []
    for auc, variance in zip(aucs, np.diag(covariance)):
        if not np.isfinite(variance) or variance < 0:
            auc_p.append(np.nan)
        elif variance == 0:
            auc_p.append(1. if auc == .5 else 0.)
        else:
            auc_p.append(2 * scipy.stats.norm.sf(abs(auc - .5) / np.sqrt(variance)))

    difference = aucs[0] - aucs[1]
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if variance == 0 and difference == 0:
        z = 0.
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.divide(difference, np.sqrt(variance))
    if alternative == "two.sided":
        p_difference = 2 * scipy.stats.norm.sf(abs(z))
    elif alternative == "greater":
        p_difference = scipy.stats.norm.sf(z)
    else:
        p_difference = scipy.stats.norm.cdf(z)

    return curves, {
        "AUC_ref": aucs[0], "AUC_p-value_ref": auc_p[0],
        "AUC_new": aucs[1], "AUC_p-value_new": auc_p[1],
        "AUC_diff_p_value": p_difference,
    }


# calculate occupied calibration bins and Brier summaries as in R
def _calibration_statistics(data_ref, data_new, n_bins=10):
    data = USbind_out(data_ref, data_new)["comparison_df"]
    y = _binary_outcomes(data["y"], both_classes=True)
    n_bins = _variable_count(n_bins, "n_bins")
    if n_bins != int(n_bins) or not 2 <= n_bins <= len(y):
        raise ValueError("n_bins must be an integer between 2 and the number of observations!")
    n_bins = int(n_bins)
    breaks = np.linspace(0, 1, n_bins + 1)
    curves = []
    scores = []
    for key in ["p_ref", "p"]:
        p = _probabilities(data[key])
        bins = np.clip(np.searchsorted(breaks, p, side="left") - 1, 0, n_bins - 1)
        occupied = np.unique(bins)
        curves.append(pd.DataFrame({
            "Predicted": [np.mean(p[bins == i]) for i in occupied],
            "Empirical": [np.mean(y[bins == i]) for i in occupied],
            "Count": [np.sum(bins == i) for i in occupied],
        }))
        scores.append(np.mean((p - y) ** 2))
    difference = scores[0] - scores[1]
    return curves, {
        "Brier_ref": scores[0], "Brier_new": scores[1],
        "Delta_Brier": difference,
        "BSS": difference / scores[0] if scores[0] > 0 else np.nan,
    }
