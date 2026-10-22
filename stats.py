import numpy as np
import pandas as pd
import scipy.stats
from prep import USprep_mdl, USbind_out


# calculate statistics for Usmile from raw data
#
# functionality of R's function of the same name
def UScalc_raw(raw_data, y_coef, n_vars_diff, bootstrap_p=None):

    if y_coef not in ["rLR", "BA", "RB"]:
        raise ValueError("y_coef must be set to rLR, BA or RB!")

    y = np.array(raw_data["y"])
    p_ref = np.array(raw_data["p_ref"])
    p = np.array(raw_data["p"])

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
    p_Y_nonev, p_Y_event, p_Y_overall = [np.nan, np.nan, np.nan]

    # relative likelihood Ratio
    if y_coef == "rLR":
        res_ref = y * np.log(p_ref + offset) + (1 - y) * np.log(1 - p_ref + offset)
        res = y * np.log(p + offset) + (1 - y) * np.log(1 - p + offset)
        
        lrt_stat = 2 * abs(np.sum(res) - np.sum(res_ref))
        p_Y_overall = scipy.stats.chi2.sf(lrt_stat, df)
        
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
    
    ref_data = USprep_mdl(ref_model, X_ref, y)
    new_data = USprep_mdl(new_model, X_new, y)

    if ref_calibrator is not None:
        ref_data["p"] = ref_calibrator.apply(ref_data["p"])
    if new_calibrator is not None:
            new_data["p"] = new_calibrator.apply(new_data["p"])

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