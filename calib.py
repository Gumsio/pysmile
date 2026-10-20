import numpy as np
from scipy.special import logit, expit
from scipy.optimize import root_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# fit and apply probability calibration
#
# functionality of R's USfit_calibrator and USapply_calibrator
class USCalibrator:
    
    def __init__(self, method="logistic", eps=1e-6):
        if method not in ["logistic", "intercept", "isotonic"]:
            raise ValueError("Wrong calibration method! Must logistic, intercept or isotonic!")
        self.method = method
        self.eps = eps
        self.model_ = None
        self.intercept_ = None


    # fit the calibration using target class and predicted probabilities
    def fit(self, y, p):
        y = np.array(y)
        p = np.array(p)

        if not np.array_equal(np.unique(y), [0, 1]): raise ValueError("Target class must be binary!")
        if np.any(p) > 1 or np.any(p) < 0: raise ValueError("Probabilities must be between 0 and 1!")

        if self.method in ["logistic", "intercept"]:
            p_restricted = np.clip(p, self.eps, 1 - self.eps)
            logit_p = logit(p_restricted)

            if self.method == "logistic":
                self.model_ = LogisticRegression(penalty=None)
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

        return self


    # apply the fitted calibration to a new vector of probabilities
    def apply(self, p):
        if (self.method in ["logistic", "isotonic"] and self.model_ is None) or (self.method == "intercept" and self.intercept_ is None):
            raise ValueError("Calibrator must be fitted before applying!")

        p = np.array(p)

        if self.method in ["logistic", "intercept"]:
            p_restricted = np.clip(p, self.eps, 1 - self.eps)
            logit_p = logit(p_restricted)

            if self.method == "logistic":
                return self.model_.predict_proba(logit_p.reshape(-1, 1))[:, 1]

            elif self.method == "intercept":
                return expit(logit_p + self.intercept_)

        elif self.method == "isotonic":
            return self.model_.predict(p)

