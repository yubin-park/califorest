import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from califorest import CaliForest
from califorest import RC30
from califorest import metrics as em

test_size = 0.3
n_estimators = 300
mss = 3
msl = 1
max_depth = 5
n_estimators = 100
random_seed = 1

np.random.seed(random_seed)
poly = PolynomialFeatures()
X, y = make_hastie_10_2(n_samples=10000)
X = poly.fit_transform(X)
y[y<0] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.3)

model = CaliForest(n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=mss,
                    min_samples_leaf=msl,
                    ctype="isotonic")

model.fit(X_train, y_train)
        
y_pred = model.predict_proba(X_test)[:,1]

score_auc = roc_auc_score(y_test, y_pred)
score_hl = em.hosmer_lemeshow(y_test, y_pred)
score_sh = em.spiegelhalter(y_test, y_pred)
score_b, score_bs = em.scaled_brier_score(y_test, y_pred)
rel_small, rel_large = em.reliability(y_test, y_pred)

print(f"AUC: {score_auc}")
print(f"Hsomer-Lemeshow statistic: {score_hl}")
print(f"Spiegelhalter z-statistic: {score_sh}")
print(f"Brier score: {score_b}")
print(f"Scaled Brier score: {score_bs}")
print(f"Reliability-in-the-small: {rel_small}")
print(f"Reliability-in-the-large: {rel_large}")
