"""Matplotlib renderers for U-SMILE results and precomputed comparison curves.

The statistical modules own all coefficient, ROC, and calibration calculations.
See docs/plotting.md for input contracts and differences from the R package.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator, MultipleLocator

from prep import USbind_out

__all__ = ["USplot", "PIWplot", "ROCplot", "CLBplot"]

_SUBCLASSES = ("nonev_be", "nonev_wo", "event_wo", "event_be")
_LABELS = (
    "non-events with better prediction",
    "non-events with worse prediction",
    "events with worse prediction",
    "events with better prediction",
)
_COLORS = ("#0F3C78", "#0F3C78", "#D51424", "#D51424")
_FILLS = ("#0F3C78", "#BED2FA", "#FBCDB9", "#D51424")
_MODEL_COLOR = "#2F4F4F"
_MM_TO_PT = 72 / 25.4


def _mapping(value, name, keys):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping containing {', '.join(keys)}.")
    missing = set(keys).difference(value)
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(sorted(missing))}.")
    return value


def _vector(value, name, *, size=None, probability=False):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty numeric vector.")
    array = array.astype(float, copy=True)
    if not np.isfinite(array).all() or (size is not None and len(array) != size):
        raise ValueError(f"{name} must contain {size or 'only'} finite values.")
    if probability and ((array < 0).any() or (array > 1).any()):
        raise ValueError(f"{name} must contain values between 0 and 1.")
    return array


def _scalar(value, name, *, missing=False, probability=False):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric scalar.")
    number = float(array)
    if missing and np.isnan(number):
        return number
    if not np.isfinite(number) or (probability and not 0 <= number <= 1):
        raise ValueError(f"{name} must be finite" + (" and between 0 and 1." if probability else "."))
    return number


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False.")
    return bool(value)


def _number(value, digits):
    return "NA" if np.isnan(value) else f"{value:.{digits}f}"


def _figure(title, *, square=False, caption=None):
    if not isinstance(title, str):
        raise ValueError("title must be a string.")
    fig = Figure(figsize=(11.5, 7), facecolor="white")
    ax = fig.subplots()
    fig.subplots_adjust(left=0.09, bottom=0.18, right=0.58, top=0.87)
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=14, pad=14)
    if not square:
        ax.title.set_position((0, 1))
        ax.title.set_ha("left")
    ax.tick_params(axis="both", which="both", length=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#DEDEDE", linewidth=0.7)
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if square:
        # ggplot coord_equal uses five percent expansion on each side.
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), aspect="equal")
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        formatter = FuncFormatter(lambda x, _: f"{x:.2f}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    if caption:
        fig.text(0.5, 0.055, caption, ha="center", va="center", color="#666666", fontsize=10)
    return fig, ax


def _legend(ax, handles, title, *, anchor=(1.04, .5), loc="center left", handleheight=.7):
    legend = ax.legend(
        handles=handles, title=title, loc=loc, bbox_to_anchor=anchor,
        frameon=False, fontsize=10, title_fontsize=12, labelspacing=1.1,
        borderaxespad=0, handleheight=handleheight,
    )
    legend.get_title().set_fontweight("bold")
    return legend


def _point_handle(label, color, fill, size=7):
    return Line2D([], [], linestyle="none", marker="o", markersize=size,
                  markeredgecolor=color, markerfacecolor=fill, label=label)


def _point_area(proportions):
    # ggplot scale_size_continuous rescales sqrt(proportion) to 0.1..10 mm.
    return ((_MM_TO_PT * (0.1 + 9.9 * np.sqrt(proportions))) ** 2)


def _us_tick(value, ticks):
    # R format() uses a common decimal precision across the whole tick vector.
    labels = [np.format_float_positional(float(f"{tick:.4g}"), trim="-") for tick in ticks]
    decimals = max((len(label.partition(".")[2]) for label in labels), default=0)
    return f"{value:.{decimals}f}"


def USplot(plot_data, y_coef, circle_sizes=True, y_lim=None, net=False, crit=0):
    """Render an existing ``UScalc_raw`` result as a U-SMILE Figure.

    ``plot_data`` also accepts ``UScalc_mdl(...)["plot_data"]`` or its full
    result. ``y_coef`` must match the coefficient used in that calculation;
    the current result structure does not carry coefficient identity.
    ``crit=2`` uses supplied class significance flags for rLR line styles.
    ``net`` displays supplied class net values. No statistics are recomputed.
    Missing overall p-values are labelled ``NA`` (including BA and RB).
    """
    if not isinstance(y_coef, str) or y_coef not in ("rLR", "BA", "RB"):
        raise ValueError("y_coef must be rLR, BA or RB.")
    circle_sizes = _boolean(circle_sizes, "circle_sizes")
    net = _boolean(net, "net")
    crit = _scalar(crit, "crit")
    if crit not in (0, 2):
        raise ValueError("crit must be 0 or 2.")
    if isinstance(plot_data, Mapping) and "plot_data" in plot_data:
        plot_data = plot_data["plot_data"]
    data = _mapping(plot_data, "plot_data", (
        "data_Y", "point_sizes", "p_Y_overall", "netto_Y_nonev", "netto_Y_event",
        "nonev_solid", "event_solid",
    ))
    table = data["data_Y"]
    if (not isinstance(table, pd.DataFrame) or table.columns.duplicated().any()
            or not {"subclass", "value", "coefficient", "segment", "wo_be"}.issubset(table.columns)):
        raise ValueError("data_Y must be a DataFrame with subclass, value, coefficient, segment and wo_be columns.")
    if len(table) != 4 or set(table["subclass"]) != set(_SUBCLASSES):
        raise ValueError("data_Y must contain each of the four subclasses exactly once.")
    ordered = table.set_index("subclass").loc[list(_SUBCLASSES)]
    if (list(ordered["segment"]) != ["nonev", "nonev", "event", "event"]
            or list(ordered["wo_be"]) != [0, 1, 1, 0]
            or not ordered["coefficient"].eq("Y").all()):
        raise ValueError("data_Y has inconsistent coefficient, segment or wo_be metadata.")
    values = _vector(ordered["value"], "data_Y.value", size=4)
    proportions = _vector(data["point_sizes"], "point_sizes", size=4, probability=True)
    p_value = _scalar(data["p_Y_overall"], "p_Y_overall", missing=True, probability=True)
    nets = [_scalar(data[key], key) for key in ("netto_Y_nonev", "netto_Y_event")]
    solid = [_boolean(data[key], key) for key in ("nonev_solid", "event_solid")]

    if y_lim is not None:
        limits = _vector(y_lim, "y_lim", size=2)
        if limits[0] >= limits[1]:
            raise ValueError("y_lim must be increasing.")
    elif y_coef == "rLR":
        limits = np.array([0., 1.])
    else:
        limits = np.array([values.min(), values.max()])
        if limits[0] == limits[1]:
            margin = max(abs(limits[0]) * 0.05, 0.01)
            limits += [-margin, margin]
    limits[1] += 0.02 * np.ptp(limits)
    span = np.ptp(limits)
    # R scale limits censor out-of-range coefficient points before drawing.
    visible = (values >= limits[0]) & (values <= limits[1])
    displayed = np.where(visible, values, np.nan)

    fig, ax = _figure(f"{y_coef} (p = {_number(p_value, 4)})")
    x = np.arange(1, 5)
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(limits[0] - 0.05 * span, limits[1] + 0.05 * span)
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(False, axis="x", which="both")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _us_tick(v, ax.get_yticks())))
    if crit == 2 and y_coef == "rLR":
        for indices, significant in (([0, 1], solid[0]), ([2, 3], solid[1])):
            ax.plot(x[indices], displayed[indices], color="black",
                    linestyle="-" if significant else ":", linewidth=0.5 * _MM_TO_PT)
        ax.plot(x[1:3], displayed[1:3], color="#7F7F7F", linestyle="-", linewidth=0.5 * _MM_TO_PT)
    else:
        ax.plot(x, displayed, color="#7F7F7F", linestyle="-", linewidth=0.5 * _MM_TO_PT)
    areas = _point_area(proportions) if circle_sizes else np.full(4, (3 * _MM_TO_PT) ** 2)
    ax.scatter(x[visible], values[visible], s=areas[visible], marker="o",
               edgecolors=np.array(_COLORS)[visible], facecolors=np.array(_FILLS)[visible],
               linewidths=1 * _MM_TO_PT, zorder=3)
    handles = [_point_handle(label, color, fill) for label, color, fill in zip(_LABELS, _COLORS, _FILLS)]
    subclass_legend = _legend(ax, handles, "Subclass", **(
        {"anchor": (1.04, .38), "loc": "upper left"} if circle_sizes else {}
    ))
    if circle_sizes:
        ax.add_artist(subclass_legend)
        breaks = np.array([0, .25, .5, .75, 1])
        handles = [_point_handle(f"{v:g}", "black", "#B3B3B3", np.sqrt(s))
                   for v, s in zip(breaks, _point_area(breaks))]
        _legend(ax, handles, "Proportion", anchor=(1.04, .96), loc="upper left", handleheight=2.5)
    if net:
        for indices, xpos, label_x, value, dark, light in (
            ([0, 1], 0.7, 1.5, nets[0], _COLORS[0], _FILLS[1]),
            ([2, 3], 4.3, 3.5, nets[1], _COLORS[3], _FILLS[2]),
        ):
            ax.plot([xpos, xpos], values[indices], color=light if value < 0 else dark, linestyle="-",
                    linewidth=2 * _MM_TO_PT, solid_capstyle="butt", clip_on=False)
            ax.text(label_x, values[indices].max() + .05 * span, f"{value:.2f}",
                    color=dark, ha="center", va="center", fontsize=4 * _MM_TO_PT,
                    clip_on=False)
    return fig


def _prepared(data, name):
    data = _mapping(data, name, ("y", "p", "n_vars"))
    y = _vector(data["y"], f"{name}.y")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}.y must be coded as 0 and 1.")
    _vector(data["p"], f"{name}.p", size=len(y), probability=True)
    if _scalar(data["n_vars"], f"{name}.n_vars") < 0:
        raise ValueError(f"{name}.n_vars must be non-negative.")
    return data


def PIWplot(data_ref=None, data_new=None, title="PIW Plot", *, comparison_df=None):
    """Compare probabilities from two ``USprep_mdl`` outputs in a Figure.

    Alternatively pass only ``comparison_df=USbind_out(...)["comparison_df"]``.
    Strict probability comparisons match ``stats.UScalc_raw``: exact ties
    are grey, and arbitrarily small nonzero changes retain their subclass.
    Only point masks are derived; no classification statistics are calculated.
    The exact-equality R definition in 3_function_PRCplot.R supplies the style;
    see docs/plotting.md for its discrepancy with 3_function_PIWplot.R.
    """
    if comparison_df is not None:
        if data_ref is not None or data_new is not None:
            raise ValueError("Supply prepared outputs or comparison_df, not both.")
        if (not isinstance(comparison_df, pd.DataFrame) or comparison_df.columns.duplicated().any()
                or not {"y", "p_ref", "p"}.issubset(comparison_df.columns)):
            raise ValueError("comparison_df must be a DataFrame containing y, p_ref and p.")
        data = comparison_df
    else:
        ref = _prepared(data_ref, "data_ref")
        new = _prepared(data_new, "data_new")
        data = USbind_out(ref, new)["comparison_df"]
    y = _vector(data["y"], "y")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must be coded as 0 and 1.")
    p_ref = _vector(data["p_ref"], "p_ref", size=len(y), probability=True)
    p = _vector(data["p"], "p", size=len(y), probability=True)
    masks = ((y == 0) & (p < p_ref), (y == 0) & (p > p_ref),
             (y == 1) & (p < p_ref), (y == 1) & (p > p_ref), p == p_ref)
    labels = (*_LABELS, "unchanged prediction")
    colors = (*_COLORS, "#7F7F7F")
    fills = (*_FILLS, "#CCCCCC")
    fig, ax = _figure(title, square=True, caption=(
        "The diagonal line represents identical predicted probabilities from the two models."
    ))
    ax.axline((0, 0), (1, 1), color="#737373", linestyle="--", linewidth=0.6 * _MM_TO_PT)
    point_colors = np.full(len(y), colors[-1], dtype="<U7")
    point_fills = np.full(len(y), fills[-1], dtype="<U7")
    for mask, color, fill in zip(masks, colors, fills):
        point_colors[mask] = color
        point_fills[mask] = fill
    # A single collection preserves R's observation order when points overlap.
    ax.scatter(p_ref, p, s=(2.5 * _MM_TO_PT) ** 2, marker="o",
               edgecolors=point_colors, facecolors=point_fills, alpha=0.7,
               linewidths=0.8 * _MM_TO_PT, zorder=3)
    ax.set_xlabel("Reference-model predicted probability", fontsize=12)
    ax.set_ylabel("New-model predicted probability", fontsize=12)
    _legend(ax, [_point_handle(label, color, fill, 4 * _MM_TO_PT)
                 for label, color, fill in zip(labels, colors, fills)], "Subclass")
    return fig


def _curve(data, name, x_key, y_key, *, roc=False):
    if isinstance(data, pd.DataFrame):
        if not {x_key, y_key}.issubset(data.columns) or data.columns.duplicated().any():
            raise ValueError(f"{name} must have unique {x_key} and {y_key} columns.")
    else:
        _mapping(data, name, (x_key, y_key))
    x = _vector(data[x_key], f"{name}.{x_key}", probability=True)
    y = _vector(data[y_key], f"{name}.{y_key}", size=len(x), probability=True)
    # geom_line sorts by x and retains order within duplicate x coordinates.
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if roc:
        starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
        lows = np.minimum.reduceat(y, starts)
        highs = np.maximum.reduceat(y, starts)
        if len(x) < 2 or (highs[:-1] > lows[1:]).any():
            raise ValueError(f"{name} must contain at least two points forming a monotone ROC curve across distinct fpr values.")
    return x, y


def _summary(results, allowed, probability_keys=()):
    if results is None:
        return {}
    if not isinstance(results, Mapping) or set(results).difference(allowed):
        raise ValueError(f"results must be a mapping with keys from: {', '.join(allowed)}.")
    return {key: _scalar(value, f"results.{key}", missing=True, probability=key in probability_keys)
            for key, value in results.items()}


def _comparison(curves, title, xlabel, ylabel, caption, *, markers=False):
    fig, ax = _figure(title, square=True, caption=caption)
    ax.axline((0, 0), (1, 1), color="#BEBEBE", linestyle="--", linewidth=.6 * _MM_TO_PT)
    for (x, y), name, style, marker in zip(curves, ("Reference", "New"), (":", "-"), ("o", "^")):
        ax.plot(x, y, color=_MODEL_COLOR, linestyle=style, linewidth=.6 * _MM_TO_PT,
                marker=marker if markers else None, markersize=2 * _MM_TO_PT,
                markeredgewidth=0, label=name)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    _legend(ax, ax.lines[1:], "Model")
    return fig


def ROCplot(curve_ref, curve_new, title="ROC Plot", *, results=None):
    """Render precomputed ROC curves; return a reusable Figure.

    Each curve is a mapping or DataFrame with probability vectors ``fpr``
    and ``tpr`` (at least two points). Points are stably sorted by fpr,
    preserving supplied order within vertical segments (including pROC's
    descending tie order). Tpr ranges must increase across distinct fpr
    values. No endpoints are added.

    Optional ``results`` accepts ``AUC_ref``, ``AUC_new``, ``AUC_p-value_ref``,
    ``AUC_p-value_new``, and ``AUC_diff_p_value``. Only the comparison p-value
    appears in the R plot title (three decimals); other summaries belong in
    the caller's results table. Missing comparison results display ``NA``.
    AUC, AUC tests and paired DeLong tests must be computed upstream.
    """
    curves = [_curve(data, name, "fpr", "tpr", roc=True)
              for data, name in ((curve_ref, "curve_ref"), (curve_new, "curve_new"))]
    keys = ("AUC_ref", "AUC_new", "AUC_p-value_ref", "AUC_p-value_new", "AUC_diff_p_value")
    summary = _summary(results, keys, probability_keys=keys)
    if not isinstance(title, str):
        raise ValueError("title must be a string.")
    plot_title = f"{title} (p = {_number(summary.get('AUC_diff_p_value', np.nan), 3)})"
    return _comparison(curves, plot_title, "1 - Specificity (False Positive Rate)",
                       "Sensitivity (True Positive Rate)",
                       "The diagonal line represents the expected performance of a non-informative classifier.")


def CLBplot(curve_ref, curve_new, title="Calibration Plot", *, results=None):
    """Render precomputed calibration summaries; return a reusable Figure.

    Curves are mappings or DataFrames with ``Predicted`` (mean probability)
    and ``Empirical`` (observed event fraction) vectors. Empty bins must be
    omitted upstream. One occupied bin is valid; NaN rows are rejected.
    Points are stably sorted by Predicted, as in R's geom_line.

    Optional ``results`` accepts precomputed ``Brier_ref``, ``Brier_new``,
    ``Delta_Brier`` and ``BSS`` (NaN allowed). As in R, these summaries do
    not appear on the plot; the caller retains them for a results table.
    No binning, Brier score, score difference, or skill score is calculated.
    """
    curves = [_curve(data, name, "Predicted", "Empirical")
              for data, name in ((curve_ref, "curve_ref"), (curve_new, "curve_new"))]
    _summary(results, ("Brier_ref", "Brier_new", "Delta_Brier", "BSS"),
             probability_keys=("Brier_ref", "Brier_new"))
    return _comparison(curves, title, "Mean predicted probability", "Observed event proportion",
                       "The diagonal line represents perfect agreement between predicted probabilities and observed event proportions.",
                       markers=True)
