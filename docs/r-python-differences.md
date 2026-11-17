# R/Python comparison summary

Reference: **Usmile 0.2.0**. The Python implementation is usable for its
implemented workflows, but **full functional and numerical parity is not
established**. Packaging does not change this status.

| Area | R Usmile | Current Python implementation |
| --- | --- | --- |
| U-SMILE statistics | rLR, BA, RB, improvement proportions and significance summaries. | Main calculations implemented. Reference fixtures agree on ordinary cases; known edge cases remain different. |
| Calibration | Logistic, intercept-only and stepwise isotonic calibration. | All three methods exist. Logistic/intercept predictions are checked against independent R outputs. Isotonic interpolation differs. |
| ROC comparison | Builds curves and returns a plot plus AUC/test results. | Public API draws supplied curves. AUC and paired DeLong calculations exist internally but are not connected to the public workflow. |
| Calibration comparison | Builds calibration bins and returns a plot plus Brier summaries. | Public API draws supplied curves. Bin and Brier calculations exist internally but are not connected to the public workflow. |
| Model preparation | R model adapters, retained training data and optional calibration. | scikit-learn-compatible probability models and pipelines, explicit evaluation data, variable metadata and optional calibration. Native R model objects are not supported. |
| U-SMILE plotting | Accepts results, raw probabilities, or formulas that fit models. | Accepts results and raw probabilities. Formula-based model fitting is absent. |
| Plot objects | ggplot objects; ROC/calibration also return result tables. | Matplotlib Figures; appearance is not pixel-identical. Some annotation and input-handling choices differ. |
| Example data | Nine bundled heart-disease datasets. | Not bundled. |
| Precision-recall | Package description mentions it, but PRCplot is not exported; its source file defines a second PIWplot. | No PRCplot is exposed; the upstream ambiguity is documented. |

Validation before packaging: **300 passing tests**, including independent R
fixtures. A separate 195-case statistical comparison matched all compared
scalar results in 171 cases; the remaining 24 concerned zero variable
differences or identical predictions. These checks do not certify every model,
input, dependency version, or plotting detail.

See [open design decisions](design-decisions.md) for choices requiring approval
and the [change report](r-alignment-report.md) for implementation details.
