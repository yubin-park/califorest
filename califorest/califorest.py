import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as Iso

class CaliForest:

    def __init__(self,
                n_estimators=300,
                criterion="gini",
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                ctype="isotonic",
                alpha0=100,
                beta0=25):

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
            self.calibrator = LR(penalty="none", 
                                solver="saga", 
                                max_iter=5000)
        elif ctype=="isotonic":
            self.calibrator = Iso(y_min=0, 
                                y_max=1,
                                out_of_bounds="clip")
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0

    def fit(self, X, y):

        n, m  = X.shape
        n_est = len(self.estimators)
        Y_oob = np.full((n, n_est), np.nan)
        n_oob = np.zeros(n)
        IB = np.zeros((n, n_est), dtype=int)
        OOB = np.full((n, n_est), True)

        for eid in range(n_est):
            IB[:,eid] = np.random.choice(n, n)
            OOB[IB[:,eid],eid] = False

        for eid, est in enumerate(self.estimators):
            ib_idx = IB[:,eid]
            oob_idx = OOB[:,eid]
            est.fit(X[ib_idx,:], y[ib_idx])
            Y_oob[oob_idx,eid] = est.predict_proba(X[oob_idx,:])[:,1]
            n_oob[oob_idx] += 1

        Y_oob_ = Y_oob[n_oob > 0,:]
        z_hat = np.nanmean(Y_oob_, axis=1)

        z_true = y[n_oob > 0]

        beta = self.beta0 + np.nanvar(Y_oob_, axis=1) * n_oob / 2
        alpha = self.alpha0 + n_oob/2
        z_weight = alpha / beta

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





