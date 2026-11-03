# Open design decisions

Status: deferred by the project owner. Packaging the current implementation
does not approve any of the changes below. Reference: R Usmile 0.2.0.

| Topic | Decision needed | Current Python behavior |
| --- | --- | --- |
| Isotonic calibration | Reproduce R's step function, duplicate-probability handling, and unusual final-knot behavior, or keep interpolation? | scikit-learn interpolation. |
| Statistical edge cases | Reproduce R's results for zero variable differences and identical predictions, or retain Python's behavior? | Some class p-values are missing instead of zero; identical predictions return zero for the overall rLR coefficient instead of an undefined value. |
| ROC/calibration interfaces | Add R-style prepared-model input and plot/table output while retaining existing calls, or replace the current API? | Public functions render precomputed curves and return a Figure; calculation helpers are internal. |
| Formula and GLM workflows | Add optional statsmodels/Patsy dependencies for R-style formula workflows, or stay with current dependencies? | Explicit model data and scikit-learn-compatible models; no formula-based fitting. |
| Example datasets | Include the nine R datasets with categories and attribution, or exclude datasets from the parity target? | R datasets are not distributed. |
| Duplicate PIW definitions | Which upstream definition is authoritative? Should Python copy R's loaded definition or preserve current rendering choices? | Uses strict ties and selected behavior documented in plotting.md. R has no exported PRCplot function. |
| Python licensing | Confirm the Python project's license and required upstream notices before sharing a release. | No project-wide Python license has been selected; no license grant is inferred by packaging. |
| Version support | Which Python/dependency versions should receive an automated test matrix? | Metadata allows Python 3.8+ based on syntax and dependency minimums; testing so far uses Python 3.14.4 only. |

Existing model-family support and plotting differences also need validation
before claiming full R parity. Statistical formulas, model training routines,
and image rendering across R and Python must not be assumed identical merely
because the installation succeeds.
