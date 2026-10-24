# pysmile

Python tools for comparing probabilistic binary classifiers with U-SMILE.

Install runtime dependencies with `python -m pip install -r requirements.txt`.
The repository currently uses flat modules; run examples from its root.

```python
from stats import UScalc_raw
from plot import USplot

# predictions is a DataFrame containing y, p_ref, and p.
result = UScalc_raw(predictions, y_coef="rLR", n_vars_diff=1)
fig = USplot(result, y_coef="rLR", net=True, crit=2)
fig.savefig("usmile.png", dpi=150)
```

`plot.py` provides `USplot`, `PIWplot`, `ROCplot`, and `CLBplot`, each returning
a reusable Matplotlib `Figure`. See [plotting interfaces and R differences](docs/plotting.md)
for usage and the inputs required for ROC and calibration integration.

Install test dependencies with `python -m pip install -r requirements-dev.txt`
and run `python -m pytest`. Tests use a noninteractive backend and checked-in
[R reference fixtures](tests/reference/README.md); R is unnecessary for normal
Python use and testing. `PRCplot` is intentionally unimplemented.
