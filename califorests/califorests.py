import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

class CaliForests:

    def __init__(self,
                n_estimators=30,
                criterion="gini",
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                ctype="isotonic"):

        estimators = []
        for i in range(n_estimators):
            estimators.append(Tree(criterion=criterion,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features="auto"))
        self.estimators = estimators
        self.calibrator = None
        if ctype=="logistic":
            self.calibrator = LR(C=1e20, solver="lbfgs")
        elif ctype=="isotonic":
            self.calibrator = IsotonicRegression(y_min=0, y_max=1,
                                    out_of_bounds="clip")
        self.ctype = ctype

    def fit(self, X, y):
        n, m  = X.shape
        n_est = len(self.estimators)
        Y_oob = np.full((n, n_est), np.nan)
        n_oob = np.zeros(n)

        for eid, est in enumerate(self.estimators):

            ib_idx = np.random.choice(n, n)
            ib_set = set(ib_idx)
            oob_idx = [rid for rid in range(n) if rid not in ib_set]
            X0, y0 = X[ib_idx,:], y[ib_idx]
            X1, y1 = X[oob_idx,:], y[oob_idx]
            est.fit(X0, y0)
            y_est = est.predict_proba(X1)[:,1]
            Y_oob[oob_idx,eid] = y_est
            n_oob[oob_idx] += 1

        thr = 0
        z_true = y[n_oob > thr]
        z_hat = np.nanmean(Y_oob[n_oob > thr,:], axis=1)
        z_std = np.nanstd(Y_oob[n_oob > thr,:], axis=1)
        z_std_mean = np.mean(z_std)
        z_weight = 1/(z_std + 1e-2 * z_std_mean)

        if self.ctype=="logistic":
            self.calibrator.fit(z_hat[:,np.newaxis], z_true, z_weight)
        elif self.ctype=="isotonic":
            self.calibrator.fit(z_hat, z_true, z_weight)

    def predict_proba(self, X):
        n, m = X.shape
        n_est = len(self.estimators)
        z = np.zeros(n)
        y_mat = np.zeros((n,2))
        for eid, est in enumerate(self.estimators):
            z += est.predict_proba(X)[:,1]
        z /= n_est

        if self.ctype=="logistic":
            y_mat[:,1] = self.calibrator.predict_proba(z[:,np.newaxis])[:,1]
        elif self.ctype=="isotonic":
            y_mat[:,1] = self.calibrator.predict(z)
        y_mat[:,0] = 1 - y_mat[:,1]
        
        return y_mat





