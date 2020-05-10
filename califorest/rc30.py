import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RC30(ClassifierMixin, BaseEstimator):

    def __init__(self, 
                n_estimators=30, 
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1, 
                ctype="isotonic"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf)
        if self.ctype == "logistic":
            self.calibrator = LogisticRegression(C=1e20, solver="lbfgs")
        elif self.ctype == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0, y_max=1,
                                                out_of_bounds="clip")
        X0, X1, y0, y1 = train_test_split(X, y, test_size=0.3) 
        self.model.fit(X0, y0)
        if self.ctype == "logistic":
            y_est = self.model.predict_proba(X1)[:,[1]]
            self.calibrator.fit(y_est, y1)
        elif self.ctype == "isotonic":
            y_est = self.model.predict_proba(X1)[:,1]
            self.calibrator.fit(y_est, y1)

        self.is_fitted_ = True
        return self
 
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        
        if self.ctype == "logistic":
            return self.calibrator.predict_proba(
                    self.model.predict_proba(X)[:,[1]])
        elif self.ctype == "isotonic":
            n, m = X.shape
            y = np.zeros((n,2))
            y[:,1] = self.calibrator.predict(
                        self.model.predict_proba(X)[:,1])
            y[:,0] = 1 - y[:,1]
            return y


