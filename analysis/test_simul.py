import csv
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from califorests import CaliForests
from califorests import RC30


n_estimators = 30
max_depth = 5
min_samples_split = 3
min_samples_leaf = 1
models = {"califorests": CaliForests(n_estimators=n_estimators,
                                    max_depth=max_depth, 
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf),
        "rf_nocal": RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf),
        "rf_cal30": RC30(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)}

def get_mortdata():
    data = pd.read_csv("data/featureSet3_48.csv")
    outcomes = pd.read_csv("data/Outcomes-a.txt")
    outcomes = outcomes[['RecordID', 'In-hospital_death']]
    data = pd.merge(data, outcomes, how='inner', on='RecordID')
    col_names = data.columns
    col_names_x = [cname for cname in col_names 
                    if cname not in ["RecordID", "Length_of_stay", 
                                        "In-hospital_death"]]
    X = data[col_names_x].values
    y = data["In-hospital_death"].values
    return X, y

#X, y = load_breast_cancer(return_X_y=True)
X, y = make_hastie_10_2(n_samples=1000)
y[y<0] = 0
#X, y = get_mortdata()
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
np.random.seed(1)

res = [["model_name", "test_size", "auc", "brier"]]
for i in range(2, 5):
    test_size = i/10.
    for j in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_pred)
            #brier = np.mean((y_test - y_pred)**2)
            row = [model_name, test_size, auc, brier]
            pprint(row)
            res.append(row)

with open("res_simul.csv", "w") as fp:
    writer = csv.writer(fp)
    writer.writerows(res)


