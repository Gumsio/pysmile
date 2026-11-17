import numpy as np
from scipy.special import logit, expit
from scipy.optimize import root_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
if __package__:
    from .prep import _binary_outcomes, _probabilities
else:
    from prep import _binary_outcomes, _probabilities


# fit and apply probability calibration
#
# functionality of R's USfit_calibrator and USapply_calibrator
class USCalibrator:
    
    def __init__(self, method="logistic", eps=1e-6):
        if not isinstance(method, str) or method not in ["logistic", "intercept", "isotonic"]:
            raise ValueError("Wrong calibration method! Must logistic, intercept or isotonic!")
        eps_value = np.asarray(eps)
        if (eps_value.ndim != 0 or eps_value.dtype.kind not in "iuf" or
                not np.isfinite(eps_value) or not 0 < eps_value < 0.5):
            raise ValueError("eps must be a finite number between 0 and 0.5!")
        self.method = method
        self.eps = eps
        self.model_ = None
        self.intercept_ = None
        self.event_level = "1"


    # fit the calibration using target class and predicted probabilities
    def fit(self, y, p):
        y = _binary_outcomes(y, both_classes=True)
        p = _probabilities(p)
        if len(y) != len(p):
            raise ValueError("Target values and probabilities must have the same length!")
        if len(y) < 3:
            raise ValueError("At least three calibration observations are required!")

        if self.method in ["logistic", "intercept"]:
            p_restricted = np.clip(p, self.eps, 1 - self.eps)
            logit_p = logit(p_restricted)

            if self.method == "logistic":
                if np.ptp(logit_p) == 0:
                    raise ValueError("Logistic calibration requires varying predicted probabilities!")
                self.model_ = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000)
                self.model_.fit(logit_p.reshape(-1, 1), y)

            elif self.method == "intercept":
                # goal function to be optimize in order to find
                # intercept alpha
                def gf(alpha):
                    return np.sum(expit(logit_p + alpha)) - np.sum(y)
                result = root_scalar(gf, bracket=[-20, 20], method="brentq")
                self.intercept_ = result.root

        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.model_.fit(p, y)

        self.n = len(y)
        self.n_non_events = int(np.sum(y == 0))
        self.n_events = int(np.sum(y == 1))
        self.prevalence = float(np.mean(y))
        self.probability_range = np.array([np.min(p), np.max(p)])
        return self


    # apply the fitted calibration to a new vector of probabilities
    def apply(self, p):
        if (self.method in ["logistic", "isotonic"] and self.model_ is None) or (self.method == "intercept" and self.intercept_ is None):
            raise ValueError("Calibrator must be fitted before applying!")

        p = _probabilities(p, allow_empty=True)
        if not p.size:
            return np.array([], dtype=float)

        if self.method in ["logistic", "intercept"]:
            p_restricted = np.clip(p, self.eps, 1 - self.eps)
            logit_p = logit(p_restricted)

            if self.method == "logistic":
                return self.model_.predict_proba(logit_p.reshape(-1, 1))[:, 1]

            elif self.method == "intercept":
                return expit(logit_p + self.intercept_)

        elif self.method == "isotonic":
            return self.model_.predict(p)


    def __repr__(self):
        lines = ["U-smile probability calibrator", f"Method: {self.method}"]
        if not hasattr(self, "n"):
            return "\n".join(lines + ["Not fitted"])
        lines.extend([f"Calibration observations: {self.n}",
                      f"Events: {self.n_events} ({100 * self.prevalence:.4g}%)"])
        if self.method == "logistic":
            lines.extend([f"Calibration intercept: {self.model_.intercept_[0]:.6g}",
                          f"Calibration slope: {self.model_.coef_[0, 0]:.6g}"])
        elif self.method == "intercept":
            lines.extend([f"Calibration intercept: {self.intercept_:.6g}",
                          "Calibration slope: 1 (fixed)"])
        else:
            lines.append(f"Isotonic knots: {len(self.model_.X_thresholds_)}")
        return "\n".join(lines)


# retain the R function names alongside the existing class interface
def USfit_calibrator(y, p, method="logistic", eps=1e-6):
    return USCalibrator(method=method, eps=eps).fit(y, p)


def USapply_calibrator(object, p):
    if not isinstance(object, USCalibrator):
        raise ValueError("object must be returned by USfit_calibrator!")
    return object.apply(p)

