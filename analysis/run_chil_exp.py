import csv
import numpy as np
import pandas as pd
from pprint import pprint
import argparse
import time
from itertools import product
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from califorest import CaliForest
from califorest import RC30
from califorest import metrics as em
import mimic_extract as mimic

def read_data(dataset, random_seed):

    X_train, X_test, y_train, y_test = None, None, None, None
    
    if dataset == "hastie":
        np.random.seed(random_seed)
        poly = PolynomialFeatures()
        X, y = make_hastie_10_2(n_samples=10000)
        X = poly.fit_transform(X)
        y[y<0] = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3)
    elif dataset == "breast_cancer":
        np.random.seed(random_seed)
        poly = PolynomialFeatures()
        X, y = load_breast_cancer(return_X_y=True)
        X = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3)
    elif dataset == "mimic3_mort_hosp":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "mort_hosp")
    elif dataset == "mimic3_mort_icu":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "mort_icu")
    elif dataset == "mimic3_los_3":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "los_3")
    elif dataset == "mimic3_los_7":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "los_7")
 
    return X_train, X_test, y_train, y_test

def init_models(n_estimators, max_depth):

    mss = 3
    msl = 1
    models = {"CF-Iso": CaliForest(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=mss,
                                        min_samples_leaf=msl,
                                        ctype="isotonic"),
            "CF-Logit": CaliForest(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=mss,
                                        min_samples_leaf=msl,
                                        ctype="logistic"),
            "RC-Iso": RC30(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=mss,
                                        min_samples_leaf=msl,
                                        ctype="isotonic"),
            "RC-Logit": RC30(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=mss,
                                        min_samples_leaf=msl,
                                        ctype="logistic"),
            "RF-NoCal": RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=mss,
                                        min_samples_leaf=msl)}
    return models


def run(dataset, random_seed):

    X_train, X_test, y_train, y_test = read_data(dataset, random_seed)

    output = []

    models = init_models(300, 10)
 
    for name, model in models.items():
        t_start = time.time()
        model.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        y_pred = model.predict_proba(X_test)[:,1]

        score_auc = roc_auc_score(y_test, y_pred)
        score_hl = em.hosmer_lemeshow(y_test, y_pred)
        score_sh = em.spiegelhalter(y_test, y_pred)
        score_b, score_bs = em.scaled_brier_score(y_test, y_pred)
        rel_small, rel_large = em.reliability(y_test, y_pred)

        row = [dataset, name, random_seed, 
               score_auc, score_b, score_bs, score_hl, score_sh,
               rel_small, rel_large] 

        print(("[info] {} {}: {:.3f} sec & BS {:.5f}").format(
                dataset, name, t_elapsed, score_b))

        output.append(row)

    return output
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    output = [["dataset", "model",
                "random_seed", "auc", "brier", "brier_scaled", 
                "hosmer_lemshow", "speigelhalter",
                "reliability_small", "reliability_large"]]

    for rs in range(10):
        output += run(args.dataset, rs)
        
    fn = "results/{}.csv".format(args.dataset)
    with open(fn, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(output)




