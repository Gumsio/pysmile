# R alignment report

Reference: Usmile 0.2.0 from the CRAN GitHub mirror. Its R functions match
upstream commit `ac0fe64bb42dc81f3112d698279e39b3a6274561`.

## Completed changes

- **Input validation:** preparation, calibration, and statistical calculations
  now reject invalid probabilities, missing values, malformed vectors, and
  misaligned observations. Statistics and calibration fitting require both
  outcome classes. Numeric, logical, and string `0`/`1` outcomes are supported.
- **Calibration:** fixed the ineffective probability-range check, validated
  the probability restriction setting, and matched R's minimum of three fit
  observations. Applying a fitted calibrator to an empty vector returns an
  empty result, as in R.
- **Model preparation:** event probabilities are selected using `classes_`,
  rather than assuming that the event always occupies the second column.
  Multiclass models, single-class models, and models without explicit class
  labels are rejected. Small rounding errors retain R's clipping tolerance.
- **Statistics:** restored the overall likelihood-test p-value for BA and RB.
  Their plot titles and model summaries now show that value. Variable counts
  and supplied bootstrap p-values are checked before calculations. Conversion
  to floating point also prevents unsigned probability arithmetic errors.
- **Tests and documentation:** added dedicated calibration, preparation, and
  statistics regression tests. Updated plotting expectations and documentation
  to reflect the restored p-values; corrected one existing indentation issue.

The flat module structure and existing positional calls are retained.
Shared internal checks live in the existing preparation module. All new
comments and documentation are in English.

## Validation

- The first correction stage passed 267 tests, including plotting and regression
  checks. Existing independently generated R fixtures verify coefficients,
  proportions, and overall p-values.
- A separate comparison covered 195 R/Python statistical cases: 171 match on
  all compared scalar results. The remaining 24 involve only zero variable
  differences or identical predictions, whose behavior is awaiting a decision.
- Editor diagnostics and the whitespace check found no errors. The temporary
  test environment reports an existing scikit-learn deprecation warning for
  `penalty=None`; this release-compatibility issue has not been changed here.

## User-visible impact

Previously accepted invalid inputs now raise explicit errors instead of
producing misleading results. Custom probability models must identify their
binary classes through `classes_`. BA/RB output now contains the R overall
likelihood-test p-value instead of a missing value. No dependency requirements
or packaging settings have changed.

## Decisions awaiting confirmation

- Whether isotonic calibration should use R's step function instead of
  the current interpolation.
- Whether to reproduce R's questionable behavior for zero variable
  differences, identical predictions, and the final isotonic knot.

These behaviors have not changed. The duplicate R PIW definitions and missing
ROC/calibration workflows remain documented gaps; no replacement API or
precision-recall behavior has been selected. Example datasets remain pending.

## Functional parity follow-up (deferred)

- **Comparison calculations:** implemented internal ROC curves, AUC,
  individual AUC tests, paired DeLong tests (including one-sided alternatives),
  occupied calibration bins, and Brier summaries. They match the existing
  independent R curves and rounded tables. Public plot integration is pending
  the decision about return types; these are not yet complete public workflows.
- **Model preparation:** added variable names and correct feature counts for
  transformed scikit-learn pipelines. `target_name` now extracts and excludes
  the outcome column from predictors. Added the R-style `dataset`/`testing`
  keywords, with explicit evaluation data required for scikit-learn models.
  Unnamed arrays retain empty variable names rather than invented labels.
- **Calibrators:** added the original R function names alongside the class,
  sample metadata, a readable printed summary, and direct application through
  model preparation without refitting. Logistic calibration now uses tighter
  convergence tolerances and rejects an unidentifiable constant predictor.
- **Plotting:** added the R raw-probability workflow to `USplot` while retaining
  its existing result-based calls. README and plotting documentation were updated.
- **Independent validation:** added a base-R calibration fixture generator and
  checked-in outputs covering duplicates, endpoints, and intermediate values.
  Logistic and intercept predictions agree within numerical tolerance. Isotonic
  fixtures are available for the compatibility decision, but its behavior has
  not been changed. Added direct mathematical checks for DeLong tests and bins.

Current validation: **300 tests pass**. Editor diagnostics and whitespace
checks are clean. The existing scikit-learn deprecation warning remains.

Additional decisions awaiting confirmation: preserving existing Figure-based
ROC/CLB calls while adding R-style plot/table returns; optional
statsmodels/Patsy support for formulas and GLMs; including the nine R example
datasets with their original names and categories. The duplicate PIW definition
is still documented, and no independent precision-recall function is present in
the reference namespace. No dependency, packaging, or publishing changes have
been made in that stage. The project owner has deferred further parity changes
to prioritize installing the current implementation with pip.

## GitHub/pip packaging (completed locally)

- Added `pyproject.toml`, a development version (`0.1.0.dev0`), automatic runtime
  dependencies, and the optional test dependencies. Runtime requirements are
  read from the existing requirements file to avoid maintaining two lists.
- Exposed the installed `pysmile` namespace and its public functions. Existing
  modules stay in place; internal imports work both inside the installed package
  and for the original flat-module usage from the repository root.
- Added source-distribution contents and ignored local build/environment files.
  The wheel includes only the library and its metadata; source archives also
  include documentation and tests. No statistical behavior has changed here.
- Updated installation instructions, development requirements, and test imports.
  Editable installation and the repository suite pass: **302 tests**.
- Created [open design decisions](design-decisions.md) and a separate
  [short R/Python comparison](r-python-differences.md), both in English.

- Built and inspected wheel and source archives. A fresh virtual environment
  installed the wheel with automatically resolved runtime/test dependencies;
  all **302 tests also pass outside the repository** against that installed
  package. Dependency checks report no broken requirements.
- Verified installation from the source archive with isolated build dependencies,
  and from a temporary local Git snapshot through pip's Git installation path.
  Checked that imports resolve to the installed library, not the checkout.
  Original flat-module calibration and plotting imports also passed a smoke test.

At the end of local packaging validation, no project commit, push, or PyPI
publication had been made. The project owner has now authorized committing and
pushing the existing `pipinstall` branch. The README uses that branch explicitly
because the default branch does not yet contain these packaging changes.
Installation from the published commit will be checked after the push.
Python 3.8+ is declared from the existing syntax and dependency minimums; only
Python 3.14.4 has been tested. Licensing and a broader support matrix remain
explicit decisions, rather than implied guarantees of the packaging change.
