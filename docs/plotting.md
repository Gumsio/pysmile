# Plotting interfaces

Import the four functions from `plot`. Each returns a Matplotlib `Figure` that
can be edited through `fig.axes`, displayed in a notebook, or saved with
`fig.savefig(...)`. Functions do not call `show`, register pyplot windows,
change global styles, fit models, apply calibrators, or mutate inputs.
The R colors, subclass order, marker families, line styles, labels, legends,
titles, captions, and decimal formatting are reproduced in Matplotlib.

The reference is [Usmile at commit ac0fe64](https://github.com/bbwieckowska/Usmile/tree/ac0fe64bb42dc81f3112d698279e39b3a6274561/R).
The [reference fixture documentation](../tests/reference/README.md) describes
the independent R executions used to validate geometry and styles.

## U-SMILE coefficients

```python
USplot(plot_data, y_coef, circle_sizes=True, y_lim=None, net=False, crit=0)
```

Supply `UScalc_raw(raw_data, y_coef, n_vars_diff, bootstrap_p=None)` directly,
`UScalc_mdl(...)["plot_data"]`, or the full `UScalc_mdl(...)` result. The
existing model interface is:

```python
UScalc_mdl(ref_model, new_model, y_coef, X_ref, X_new=None, y=None,
           ref_calibrator=None, new_calibrator=None)
```

`y_coef` must be `rLR`, `BA`, or `RB` and must match the earlier calculation.
The current statistical result does not record its coefficient name, so the
renderer cannot detect a mismatch. No raw-data or model-fitting workflow is
added to `USplot`; call the existing statistical functions first.

The renderer uses these existing keys:

| Key | Purpose |
| --- | --- |
| `data_Y` | DataFrame with `subclass`, `value`, `coefficient`, `segment`, `wo_be` |
| `point_sizes` | Four within-outcome subclass proportions in canonical order |
| `netto_Y_nonev`, `netto_Y_event` | Supplied net coefficients and their labels |
| `nonev_solid`, `event_solid` | Supplied significance flags for `crit=2` with rLR |
| `p_Y_overall` | Overall title p-value, formatted to four decimal places |

Rows are displayed in this fixed order: `nonev_be`, `nonev_wo`, `event_wo`,
`event_be`. Shuffling `data_Y` does not change point-size associations.
The scalar `Y_*` entries duplicate the table's values and are not recomputed
or needed by rendering. Net labels always use supplied net statistics.

`crit=0` connects all points with a solid grey line. `crit=2` with rLR uses
black solid or dotted lines for the non-event and event pairs according to
the corresponding significance flags, plus a grey middle connector.
BA/RB always use the solid grey connector. Proportional circle diameters
follow R's `0.1 + 9.9 * sqrt(proportion)` size scale in millimetres; setting
`circle_sizes=False` uses constant 3 mm circles and hides the size legend.

The default rLR scale starts from `[0, 1]`. BA/RB use the finite coefficient
range, including values greater than 1 for RB, and expand equal ranges.
As in R, `y_lim` supplies scale limits: the upper limit gains 2% of the
span and the viewing limits gain a further 5% margin on each side.
Coefficient geometry outside the scale limits is omitted. Net annotations
remain visible; this avoids R's omission of labels above the BA/RB scale.
Axis numbers use four significant digits without scientific notation;
coefficients are numeric rather than percentages. Net labels use two decimals.

## Prediction improvement/worsening

```python
PIWplot(data_ref=None, data_new=None, title="PIW Plot", *, comparison_df=None)
```

Use two outputs of `USprep_mdl(model, X, y=None, target_name=None)`:

```python
from prep import USprep_mdl, USbind_out
from plot import PIWplot

ref = USprep_mdl(ref_model, X_ref, y)
new = USprep_mdl(new_model, X_new, y)
fig = PIWplot(ref, new)

# An existing comparison frame can be reused directly.
bound = USbind_out(ref, new)
same_plot = PIWplot(comparison_df=bound["comparison_df"])
```

Prepared outputs contain `y`, `p`, and `n_vars`. Pair validation and combining
are delegated to `USbind_out`. A supplied `comparison_df` must contain
`y`, `p_ref`, and `p`. Outcomes must be numeric or boolean 0/1, probabilities
finite and in `[0, 1]`, and all vectors nonempty and aligned. Single-class
inputs are supported. Apply an existing `USCalibrator` upstream if needed.

`stats.py` does not export a per-observation subclass assignment helper.
Consequently PIW derives only the four drawing masks using its exact same
strict comparisons: a lower probability improves non-events, and a higher
probability improves events. Exact ties appear grey. No coefficients,
proportions, thresholds, accuracy scores, or significance tests are recalculated.
The tests cross-check point counts against existing `UScalc_raw` proportions.

### Duplicate R definitions

The pinned R source defines `PIWplot` in **both** `3_function_PIWplot.R` and
`3_function_PRCplot.R`. These are different implementations:

| Behavior | PIW file | PRC file's duplicate PIW |
| --- | --- | --- |
| Unchanged predictions | `abs(delta) <= sqrt(.Machine$double.eps)` | `delta == 0` |
| Unchanged border | `grey55` | `grey50` |
| Both models have zero variables | Text-only placeholder | Normal scatter plot |
| Single outcome class | Accepted | Rejected |
| Outcome encodings | Numeric/logical | Also character/factor |
| Axis labels/caption | Short probability wording | Explicit predicted probability wording |

Python follows the **PRC file's duplicate** for exact comparisons, colors,
labels, caption, and normal plotting of zero-variable models. This matches
the current Python statistical semantics. It accepts numeric/logical outcomes
and single-class inputs consistently with Python's statistics and the primary
R PIW definition. Near-zero changes are not assigned a tolerance. Grey points
correspond to `stats.py`'s unassigned (`unknown`) observations. No `PRCplot`
function is introduced.

## ROC curves

```python
ROCplot(curve_ref, curve_new, title="ROC Plot", *, results=None)
```

Each curve is a mapping or DataFrame containing `fpr` and `tpr` probability
vectors with equal length and at least two points. Coordinates must lie in
`[0, 1]`. Points are stably sorted by `fpr`, matching R's `geom_line`; order
within duplicate false-positive rates is preserved, including descending
vertical segments produced by pROC. TPR ranges must not decrease between
distinct FPR groups. Supplied endpoints are preserved, and none are added.
The plot connects the supplied coordinates with straight segments as R does.

```python
# ref_curve and new_curve are supplied by the future ROC statistics layer.
fig = ROCplot(ref_curve, new_curve, results=roc_results)
```

`results` is an optional numeric mapping with any of these keys:
`AUC_ref`, `AUC_p-value_ref`, `AUC_new`, `AUC_p-value_new`,
`AUC_diff_p_value`. Values must be in `[0, 1]` or NaN. Only
`AUC_diff_p_value` appears in the title, to three decimals. An omitted or NaN
comparison p-value appears as `p = NA`. Unknown keys are rejected to catch
misspellings. AUC estimates and individual AUC p-values remain in the caller's
summary table, matching R's separation of plot and table.

Future integration must supply ROC points with larger event probabilities
representing the positive class, AUC estimates, tests of AUC against 0.5,
and the paired DeLong comparison with an explicit alternative. The R
implementation supplies Reference first and New second to its paired test.
The renderer does not infer the test alternative or calculate any ROC
statistics. R's column names map to Python as follows:

| R curve column | Python curve column |
| --- | --- |
| `FalsePositiveRate` | `fpr` |
| `Sensitivity` | `tpr` |

## Calibration curves

```python
CLBplot(curve_ref, curve_new, title="Calibration Plot", *, results=None)
```

Each curve is a mapping or DataFrame with `Predicted` (mean predicted
probability) and `Empirical` (observed event proportion) vectors. Values
must be finite and in `[0, 1]`. Empty bins must be omitted upstream rather
than represented by NaNs. One occupied bin is valid. Additional R summary
columns such as `bin`, `Count`, and `Model` are accepted and ignored.
Points are stably sorted by `Predicted`, with no rebinning or smoothing.

`results` optionally accepts numeric `Brier_ref`, `Brier_new`, `Delta_Brier`,
and `BSS`. Brier scores must be in `[0, 1]` or NaN; differences and skill
scores may be negative, and NaN BSS is accepted for a perfect reference
model. Summaries are validated but not drawn, matching R; retain the
original mapping for a results table. No missing summary is inferred.

Future integration must supply occupied-bin summaries and any desired
Brier statistics. To match R, the upstream binning should use equal-width,
right-closed intervals over `[0, 1]`, include zero in the lowest interval,
and omit empty bins. R reports `Delta_Brier = Brier_ref - Brier_new` and
`BSS = Delta_Brier / Brier_ref`, with missing BSS when the reference score
is zero. These formulas document the required upstream semantics; they
are not implemented in plotting. `USCalibrator` fits/applies probabilities
and does not currently produce these calibration summaries.

## Confirmed differences and limits

- Python returns `Figure` for all four functions. R returns ggplot objects
  for US/PIW and lists containing plots and statistical tables for ROC/CLB.
  Summaries stay with the Python caller; ROC/CLB rendering computes no statistics.
- Existing Python BA/RB results have `p_Y_overall = NaN`; R calculates the
  overall likelihood-test p-value even for these coefficients. Python displays
  `NA` and leaves statistical semantics unchanged. Supplying a precomputed
  finite p-value displays it without recalculation.
- PIW resolves the duplicate definitions as described above. Single-class
  numeric data remain supported and zero-variable models remain visible.
- US net labels are retained outside the coefficient scale, whereas R can
  censor these annotations for BA/RB or a narrow custom range.
- Matplotlib font metrics, tick placement, legend spacing, marker stroke
  geometry, and antialiasing are backend dependent. Sizes use the R scale
  converted from millimetres to points; screenshots are not pixel identical.
  Axis values and title/net decimal precision are preserved.
- Inputs are validated before rendering. ROC/CLB reject missing coordinates;
  empty calibration bins must be removed upstream. The R plotting layers
  may instead omit missing rows while issuing a warning.

`stats.py`, `prep.py`, and `calib.py` are unchanged. The plotting tests cover
all coefficients, actual fitted-model results, edge cases, input validation,
non-mutation, vector export, and independently generated R geometry/styles.
No upstream statistical parity beyond these fixtures is claimed.
