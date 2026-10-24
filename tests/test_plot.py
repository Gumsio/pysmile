"""Plot geometry, contracts, and integration with the existing statistics."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path

import matplotlib.colors as colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from plot import CLBplot, PIWplot, ROCplot, USplot
from prep import USbind_out, USprep_mdl
from stats import UScalc_mdl, UScalc_raw


REFERENCE = Path(__file__).parent / "reference"
COEFFICIENTS = ("rLR", "BA", "RB")
SUBCLASSES = ("nonev_be", "nonev_wo", "event_wo", "event_be")


def fixture_table(name):
    return pd.read_csv(REFERENCE / name, sep="\t")


@pytest.fixture
def predictions():
    return fixture_table("predictions.tsv")


@pytest.fixture
def prepared(predictions):
    return ({"y": predictions.y.to_numpy(), "p": predictions.p_ref.to_numpy(), "n_vars": 1},
            {"y": predictions.y.to_numpy(), "p": predictions.p.to_numpy(), "n_vars": 2})


@pytest.fixture
def roc_curves():
    data = fixture_table("roc_curves.tsv")
    return tuple(data.loc[data.Model == name].rename(columns={
        "FalsePositiveRate": "fpr", "Sensitivity": "tpr"
    }) for name in ("Reference", "New"))


@pytest.fixture
def calibration_curves():
    data = fixture_table("calibration_curves.tsv")
    return tuple(data.loc[data.Model == name] for name in ("Reference", "New"))


def assert_unchanged(left, right):
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_unchanged(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_unchanged(a, b)
    else:
        np.testing.assert_equal(left, right)


def assert_renderable(fig):
    assert isinstance(fig, Figure)
    FigureCanvasAgg(fig).draw()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=60)
    assert buffer.getvalue().startswith(b"\x89PNG")


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
@pytest.mark.parametrize("circle_sizes, net, crit", [(True, False, 0), (False, True, 2)])
def test_usplot_existing_raw_results(predictions, coefficient, circle_sizes, net, crit):
    original_predictions = predictions.copy(deep=True)
    result = UScalc_raw(predictions, coefficient, 1)
    before = deepcopy(result)
    fig = USplot(result, coefficient, circle_sizes=circle_sizes, net=net, crit=crit)
    ax = fig.axes[0]
    points = ax.collections[0]
    np.testing.assert_allclose(points.get_offsets(), np.column_stack([
        [1, 2, 3, 4], result["data_Y"].value
    ]))
    assert [text.get_text() for text in ax.texts] == (
        [f"{result['netto_Y_nonev']:.2f}", f"{result['netto_Y_event']:.2f}"] if net else []
    )
    if coefficient != "rLR":
        assert ax.get_title() == f"{coefficient} (p = NA)"
        assert ax.lines[0].get_color() == "#7F7F7F"
    else:
        assert ax.get_title() == f"rLR (p = {result['p_Y_overall']:.4f})"
    if not circle_sizes:
        np.testing.assert_allclose(points.get_sizes(), (3 * 72 / 25.4) ** 2)
    assert_renderable(fig)
    assert_unchanged(result, before)
    assert_unchanged(predictions, original_predictions)


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_usplot_fitted_model_integration(coefficient):
    rng = np.random.default_rng(24)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + .5 * X[:, 1] + rng.normal(size=40) > 0).astype(int)
    ref = LogisticRegression().fit(X[:, :1], y)
    new = LogisticRegression().fit(X, y)
    result = UScalc_mdl(ref, new, coefficient, X[:, :1], X, y)
    before = deepcopy(result)
    direct = USplot(result["plot_data"], coefficient)
    wrapped = USplot(result, coefficient)
    np.testing.assert_array_equal(direct.axes[0].collections[0].get_offsets(),
                                  wrapped.axes[0].collections[0].get_offsets())
    assert_unchanged(result, before)
    prepared_ref = USprep_mdl(ref, X[:, :1], y)
    prepared_new = USprep_mdl(new, X, y)
    assert_renderable(PIWplot(prepared_ref, prepared_new))


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_usplot_r_reference_points(predictions, coefficient):
    # These coordinates and styles came from ggplot_build, not Python.
    reference = fixture_table(f"us_{coefficient}_points.tsv")
    result = UScalc_raw(predictions, coefficient, 1)
    points = USplot(result, coefficient).axes[0].collections[0]
    np.testing.assert_allclose(points.get_offsets(), reference[["x", "y"]], atol=1e-14)
    np.testing.assert_allclose(points.get_facecolors(), [colors.to_rgba(c) for c in reference.fill])
    np.testing.assert_allclose(points.get_edgecolors(), [colors.to_rgba(c) for c in reference.colour])
    np.testing.assert_allclose(np.sqrt(points.get_sizes()) / (72 / 25.4), reference["size"])


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_usplot_accepts_independent_r_results(coefficient):
    values = fixture_table(f"us_{coefficient}_values.tsv")
    summary = fixture_table(f"us_{coefficient}_summary.tsv").iloc[0].to_dict()
    summary.update({
        "data_Y": values[["subclass", "value"]].assign(
            coefficient="Y", segment=["nonev", "nonev", "event", "event"], wo_be=[0, 1, 1, 0]),
        "point_sizes": values.proportion.to_numpy(),
    })
    fig = USplot(summary, coefficient, crit=2, net=True)
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets()[:, 1], values.value)
    assert "NA" not in fig.axes[0].get_title()


def test_usplot_order_flags_and_supplied_net_values(predictions):
    result = UScalc_raw(predictions, "rLR", 1)
    expected = result["data_Y"].value.to_numpy().copy()
    result["data_Y"] = result["data_Y"].iloc[[3, 1, 0, 2]]
    result["point_sizes"] = [0, .25, .5, 1]
    result.update(nonev_solid=True, event_solid=False, netto_Y_nonev=-0.1234, netto_Y_event=0.2345)
    ax = USplot(result, "rLR", crit=2, net=True).axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], expected)
    np.testing.assert_allclose(np.sqrt(ax.collections[0].get_sizes()),
                               (0.1 + 9.9 * np.sqrt(result["point_sizes"])) * 72 / 25.4)
    assert [line.get_linestyle() for line in ax.lines[:3]] == ["-", ":", "-"]
    assert [text.get_text() for text in ax.texts] == ["-0.12", "0.23"]
    assert [line.get_color() for line in ax.lines[3:]] == ["#BED2FA", "#D51424"]


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_usplot_all_predictions_tied(coefficient):
    data = pd.DataFrame({"y": [0, 0, 1, 1], "p_ref": [.2, .3, .6, .7], "p": [.2, .3, .6, .7]})
    ax = USplot(UScalc_raw(data, coefficient, 1), coefficient, net=True).axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], 0)
    assert ax.get_ylim()[1] > ax.get_ylim()[0]


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_usplot_probability_endpoints_and_empty_subclasses(coefficient):
    data = pd.DataFrame({"y": [0, 0, 1, 1], "p_ref": [.5] * 4, "p": [0, 0, 1, 1]})
    ax = USplot(UScalc_raw(data, coefficient, 1), coefficient, net=True).axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[[1, 2], 1], 0)


def test_usplot_preserves_unbounded_rb_values_and_custom_limits():
    data = pd.DataFrame({"y": [0, 0, 1, 1], "p_ref": [.01, .01, .99, .99], "p": [.9, .9, .1, .1]})
    result = UScalc_raw(data, "RB", 1)
    ax = USplot(result, "RB").axes[0]
    assert ax.collections[0].get_offsets()[:, 1].max() > 1000
    assert ax.get_ylim()[1] > 1000
    clipped = USplot(result, "RB", y_lim=[0, 1]).axes[0]
    assert len(clipped.collections[0].get_offsets()) == 2
    np.testing.assert_allclose(clipped.get_ylim(), [-.051, 1.071])


@pytest.mark.parametrize("kwargs", [
    {"y_coef": "I"}, {"y_coef": None}, {"y_coef": ["BA"]},
    {"crit": 1}, {"crit": True}, {"crit": np.nan},
    {"net": 1}, {"circle_sizes": None}, {"y_lim": [1, 0]},
    {"y_lim": [0, np.inf]}, {"y_lim": [1]}, {"y_lim": [0, 0]},
])
def test_usplot_invalid_options(predictions, kwargs):
    result = UScalc_raw(predictions, "rLR", 1)
    options = {"y_coef": "rLR", **kwargs}
    with pytest.raises(ValueError):
        USplot(result, **options)


@pytest.mark.parametrize("key, value", [
    ("data_Y", None), ("point_sizes", [0, 1]),
    ("point_sizes", [-1, 0, 0, 0]), ("point_sizes", [0, 0, 0, np.nan]),
    ("point_sizes", [0, 0, 0, 2]), ("p_Y_overall", np.inf),
    ("p_Y_overall", 1.1), ("nonev_solid", None), ("event_solid", "False"),
    ("netto_Y_event", np.nan),
])
def test_usplot_invalid_result_fields(predictions, key, value):
    result = UScalc_raw(predictions, "rLR", 1)
    result[key] = value
    with pytest.raises(ValueError):
        USplot(result, "rLR")


@pytest.mark.parametrize("field, values", [
    ("subclass", ["nonev_be"] * 4), ("value", [0, 1, 2, np.nan]),
    ("segment", ["event"] * 4), ("wo_be", [0] * 4), ("coefficient", ["BA"] * 4),
])
def test_usplot_invalid_table_metadata(predictions, field, values):
    result = UScalc_raw(predictions, "BA", 1)
    result["data_Y"][field] = values
    with pytest.raises(ValueError):
        USplot(result, "BA")


@pytest.mark.parametrize("data", [None, {}, [], {"plot_data": None}])
def test_usplot_missing_result(data):
    with pytest.raises(ValueError):
        USplot(data, "rLR")


def test_piw_existing_prep_outputs_and_bound_frame(prepared):
    before = deepcopy(prepared)
    bound = USbind_out(*prepared)
    bound_before = deepcopy(bound)
    direct = PIWplot(*prepared)
    frame = PIWplot(comparison_df=bound["comparison_df"])
    for a, b in zip(direct.axes[0].collections, frame.axes[0].collections):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    assert_renderable(direct)
    assert_unchanged(prepared, before)
    assert_unchanged(bound, bound_before)


def test_piw_subclasses_agree_with_stats_including_tiny_changes(predictions, prepared):
    ax = PIWplot(*prepared).axes[0]
    point_colors = ax.collections[0].get_facecolors()
    counts = np.array([np.sum(np.all(np.isclose(point_colors, colors.to_rgba(fill, alpha=.7)), axis=1))
                       for fill in ("#0F3C78", "#BED2FA", "#FBCDB9", "#D51424")])
    result = UScalc_raw(predictions, "BA", 1)
    expected = np.array(result["point_sizes"]) * [result["n0"], result["n0"], result["n1"], result["n1"]]
    np.testing.assert_array_equal(counts, expected)
    assert np.sum(np.all(np.isclose(point_colors, colors.to_rgba("#CCCCCC", alpha=.7)), axis=1)) == 2
    assert len(ax.get_legend().get_texts()) == 5


def test_piw_r_reference_styles_and_duplicate_discrepancy(prepared):
    ax = PIWplot(*prepared).axes[0]
    reference = fixture_table("piw_exact_points.tsv")
    tolerance = fixture_table("piw_tolerance_points.tsv")
    assert sum(reference.colour == "grey50") == 2
    assert sum(tolerance.colour == "grey55") == 4
    points = ax.collections[0]
    # Preserve original observation order as well as colors, including overlaps.
    np.testing.assert_allclose(points.get_offsets(), reference[["x", "y"]], atol=1e-15)
    np.testing.assert_allclose(points.get_facecolors(), [
        colors.to_rgba("#CCCCCC" if fill == "grey80" else fill, alpha=.7) for fill in reference.fill
    ])
    np.testing.assert_allclose(points.get_edgecolors(), [
        colors.to_rgba("#7F7F7F" if color == "grey50" else color, alpha=.7) for color in reference.colour
    ])
    assert ax.get_xlabel() == "Reference-model predicted probability"
    assert ax.get_ylabel() == "New-model predicted probability"
    assert ax.get_aspect() == 1


@pytest.mark.parametrize("outcome", [0, 1])
@pytest.mark.parametrize("n_vars", [0, 1])
def test_piw_single_class_empty_models_and_ties(outcome, n_vars):
    data = {"y": [outcome, outcome], "p": [0., 1.], "n_vars": n_vars}
    ax = PIWplot(data, data).axes[0]
    assert len(ax.collections[0].get_offsets()) == 2
    np.testing.assert_allclose(ax.collections[0].get_facecolors(),
                               [colors.to_rgba("#CCCCCC", alpha=.7)] * 2)


@pytest.mark.parametrize("field, value", [
    ("y", [0, 2]), ("y", ["0", "1"]), ("y", []), ("y", [np.nan]),
    ("p", [0, 1.1]), ("p", [0, -.1]), ("p", [np.nan]), ("p", [[.1]]),
    ("p", [.1]), ("n_vars", -1), ("n_vars", np.nan), ("n_vars", None),
])
def test_piw_invalid_prepared_outputs(field, value):
    ref = {"y": [0, 1], "p": [.1, .9], "n_vars": 1}
    new = {**ref, field: value}
    with pytest.raises(ValueError):
        PIWplot(ref, new)


def test_piw_misaligned_outcomes_ambiguous_inputs_and_missing_fields(prepared, predictions):
    new = deepcopy(prepared[1])
    new["y"] = 1 - new["y"]
    with pytest.raises(ValueError, match="Target values differ"):
        PIWplot(prepared[0], new)
    with pytest.raises(ValueError, match="not both"):
        PIWplot(*prepared, comparison_df=predictions)
    for ref, new in (({}, {}), (None, None), (prepared[0], {})):
        with pytest.raises(ValueError):
            PIWplot(ref, new)
    with pytest.raises(ValueError):
        PIWplot(comparison_df=predictions.drop(columns="p"))


@pytest.mark.parametrize("name", ["roc", "calibration"])
def test_comparison_r_reference_geometry_and_nonmutation(name, roc_curves, calibration_curves):
    curves = roc_curves if name == "roc" else calibration_curves
    plot = ROCplot if name == "roc" else CLBplot
    before = deepcopy(curves)
    results = fixture_table(f"{name}_results.tsv").set_index("Metric").Value.to_dict()
    results_before = deepcopy(results)
    fig = plot(*curves, results=results)
    ax = fig.axes[0]
    reference = fixture_table(f"{name}_lines.tsv")
    for line, group, style in zip(ax.lines[1:], (1, 2), (":", "-")):
        expected = reference[reference.group == group]
        np.testing.assert_allclose(line.get_xydata(), expected[["x", "y"]], atol=1e-15)
        assert line.get_color() == expected.colour.iloc[0]
        assert line.get_linestyle() == style
    if name == "roc":
        assert ax.get_title() == "ROC Plot (p = 0.480)"
        assert [line.get_marker() for line in ax.lines[1:]] == ["None", "None"]
    else:
        assert ax.get_title() == "Calibration Plot"
        assert [line.get_marker() for line in ax.lines[1:]] == ["o", "^"]
        assert len(ax.texts) == 0  # R does not overlay Brier summaries.
    assert ax.get_aspect() == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Reference", "New"]
    assert_renderable(fig)
    assert_unchanged(curves, before)
    assert_unchanged(results, results_before)


def test_roc_no_invented_results_endpoints_or_interpolation():
    ref = {"fpr": [.2, .5], "tpr": [.4, .8]}
    ax = ROCplot(ref, ref).axes[0]
    assert ax.get_title() == "ROC Plot (p = NA)"
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [.2, .5])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [.4, .8])
    assert ax.lines[1].get_drawstyle() == "default"


def test_roc_vertical_segments_in_both_pROC_and_ascending_order():
    ascending = {"fpr": [0, 0, .5, 1], "tpr": [0, .4, .8, 1]}
    descending = {"fpr": [1, .5, 0, 0], "tpr": [1, .8, .4, 0]}
    ax = ROCplot(ascending, descending, results={"AUC_diff_p_value": 0}).axes[0]
    np.testing.assert_array_equal(ax.lines[2].get_ydata(), [.4, 0, .8, 1])
    assert ax.get_title() == "ROC Plot (p = 0.000)"


def test_calibration_single_occupied_bin_and_no_binning():
    curve = {"Predicted": [.4], "Empirical": [1.]}
    ax = CLBplot(curve, curve, results={"Brier_ref": 0, "BSS": np.nan}).axes[0]
    np.testing.assert_array_equal(ax.lines[1].get_xydata(), [[.4, 1.]])
    assert_renderable(ax.figure)


@pytest.mark.parametrize("plot, keys", [(ROCplot, ("fpr", "tpr")), (CLBplot, ("Predicted", "Empirical"))])
@pytest.mark.parametrize("x, y", [([], []), ([0, 1], [.2]), ([-.1, 1], [0, 1]),
                                    ([0, np.nan], [0, 1]), ([0, 1], [0, np.inf]),
                                    ([0, 1], [0, 1.1]), ([[0, 1]], [[0, 1]])])
def test_comparison_invalid_curves(plot, keys, x, y):
    data = dict(zip(keys, (x, y)))
    with pytest.raises(ValueError):
        plot(data, data)


@pytest.mark.parametrize("data", [None, {}, {"fpr": [0], "tpr": [0]}, {"fpr": [0, 1], "tpr": [1, 0]}])
def test_roc_missing_or_nonmonotone_curves(data):
    with pytest.raises(ValueError):
        ROCplot(data, data)


@pytest.mark.parametrize("plot, curves_fixture", [(ROCplot, "roc_curves"), (CLBplot, "calibration_curves")])
@pytest.mark.parametrize("results", [{"unknown": 1}, {"AUC_diff_p_value": "0.1"}, []])
def test_comparison_invalid_summaries(request, plot, curves_fixture, results):
    with pytest.raises(ValueError):
        plot(*request.getfixturevalue(curves_fixture), results=results)


def test_comparison_invalid_summary_ranges(roc_curves, calibration_curves):
    for results in ({"AUC_ref": 1.1}, {"AUC_diff_p_value": -.1}, {"AUC_new": np.inf}):
        with pytest.raises(ValueError):
            ROCplot(*roc_curves, results=results)
    with pytest.raises(ValueError):
        CLBplot(*calibration_curves, results={"Brier_ref": -1})


def test_figures_do_not_change_pyplot_or_global_style(prepared, roc_curves, calibration_curves, predictions):
    numbers = plt.get_fignums()
    style = dict(plt.rcParams)
    for fig in (PIWplot(*prepared), ROCplot(*roc_curves), CLBplot(*calibration_curves),
                USplot(UScalc_raw(predictions, "BA", 1), "BA")):
        assert isinstance(fig, Figure)
    assert plt.get_fignums() == numbers
    assert dict(plt.rcParams) == style


@pytest.mark.parametrize("plot, curves_fixture", [(PIWplot, "prepared"), (ROCplot, "roc_curves"), (CLBplot, "calibration_curves")])
def test_invalid_titles(request, plot, curves_fixture):
    with pytest.raises(ValueError, match="title"):
        plot(*request.getfixturevalue(curves_fixture), title=None)


def test_duplicate_columns_rejected(predictions, roc_curves):
    with pytest.raises(ValueError):
        PIWplot(comparison_df=pd.concat([predictions, predictions[["y"]]], axis=1))
    with pytest.raises(ValueError):
        ROCplot(pd.concat([roc_curves[0], roc_curves[0][["fpr"]]], axis=1), roc_curves[1])


@pytest.mark.parametrize("format", ["svg", "pdf"])
def test_vector_export(roc_curves, format):
    buffer = BytesIO()
    ROCplot(*roc_curves).savefig(buffer, format=format)
    assert len(buffer.getvalue()) > 1000
