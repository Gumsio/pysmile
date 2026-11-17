# pysmile

Python tools for comparing probabilistic binary classifiers with U-SMILE.

Install the development package from the `pipinstall` branch:

```sh
python -m pip install "git+https://github.com/Gumsio/pysmile.git@pipinstall"
```

Git must be installed. To reproduce a specific revision, append `@<commit-or-tag>`
instead of `@pipinstall`. The unqualified repository URL can be used once
packaging is merged into the default branch. No PyPI publication or R installation is required.
Runtime dependencies are installed automatically from PyPI.

From a local checkout, install with `python -m pip install .`. The installed
package can be imported from any working directory. Python 3.8+ is declared
from the current syntax and dependency minimums; verification so far uses
Python 3.14.4, rather than a full Python-version support matrix.

```python
from pysmile import UScalc_raw, USplot

# predictions is a DataFrame containing y, p_ref, and p.
result = UScalc_raw(predictions, y_coef="rLR", n_vars_diff=1)
fig = USplot(result, y_coef="rLR", net=True, crit=2)
fig.savefig("usmile.png", dpi=150)
```

`pysmile.plot` provides `USplot`, `PIWplot`, `ROCplot`, and `CLBplot`, each returning
a reusable Matplotlib `Figure`. See [plotting interfaces and R differences](docs/plotting.md)
for usage and the inputs required for ROC and calibration integration.

The R names `USfit_calibrator` and `USapply_calibrator` are available from
`pysmile.calib` alongside the existing `USCalibrator` class. These public names
can also be imported directly from `pysmile`. Prepared model outputs
include `var_names`; `USbind_out` also returns `ref_vars` and `new_vars`.
`USprep_mdl` accepts an independent `calibrator` and can extract outcomes
from a DataFrame column selected through `target_name`.

`USplot(raw_data=predictions, y_coef="rLR", n_vars_diff=1)` also supports
R's raw-probability workflow. See the [R alignment report](docs/r-alignment-report.md)
for completed work, validation, and outstanding compatibility decisions.

For development, run `python -m pip install -e ".[test]"` from the repository
root, then `python -m pytest`. The equivalent requirements-file command is
`python -m pip install -r requirements-dev.txt`. Tests use a noninteractive backend and checked-in
[R reference fixtures](tests/reference/README.md); R is unnecessary for normal
Python use and testing. `PRCplot` is intentionally unimplemented.

This is a development package (`0.1.0.dev0`), not a claim of full parity with
R Usmile. See the [short R/Python comparison](docs/r-python-differences.md)
and [open design decisions](docs/design-decisions.md) for remaining differences.
