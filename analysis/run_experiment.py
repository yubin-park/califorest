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
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from califorests import CaliForests
from califorests import RC30
from califorests import eval_metrics as em
import mimic_extract as mimic

def read_data(dataset, random_seed):

    X_train, X_test, y_train, y_test = None, None, None, None
    
    if dataset == "hastie":
        np.random.seed(random_seed)
        X, y = make_hastie_10_2(n_samples=1000)
        y[y<0] = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3)
    elif dataset == "breast_cancer":
        np.random.seed(random_seed)
        X, y = make_hastie_10_2(n_samples=1000)
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3)
    elif dataset == "mimic3_mort_hosp":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "mosrt_hosp")
    elif dataset == "mimic3_mort_icu":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "mosrt_hosp")
    elif dataset == "mimic3_los3":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "los3")
    elif dataset == "mimic3_los7":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, 
                                                        "los7")
 
    return X_train, X_test, y_train, y_test

def init_models(n_estimators, max_depth):

    mss = 3
    msl = 1
    models = {"CF-Iso": CaliForests(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=mss,
                                        min_samples_leaf=msl,
                                        ctype="isotonic"),
            "CF-Logit": CaliForests(n_estimators=n_estimators,
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

    nest_lst = [30, 100, 200]
    mdep_lst = [3, 5, 7]

    output = []
    for nest, mdep in product(nest_lst, mdep_lst):
    
        models = init_models(nest, mdep)
     
        for name, model in models.items():
            t_start = time.time()
            model.fit(X_train, y_train)
            t_elapsed = time.time() - t_start
            print("[info] {} {} {} {}: {:.3f} sec".format(dataset,
                        nest, mdep, name, t_elapsed))

            y_pred = model.predict_proba(X_test)[:,1]

            score_auc = roc_auc_score(y_test, y_pred)
            score_hl = em.hosmer_lemeshow(y_test, y_pred)
            score_sh = em.spiegelhalter(y_test, y_pred)
            score_b, score_bs = em.scaled_Brier(y_test, y_pred)

            row = [dataset, name, nest, mdep, random_seed, 
                   score_auc, score_b, score_bs, score_hl, score_sh] 
            output.append(row)

    return output
       

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    output = [["dataset", "model", "n_estimators", "max_depth", 
                "random_seed", "auc", "brier", "brier_scaled", 
                "hosmer_lemshow", "speigelhalter"]]
    for rs in range(10):
        output += run(args.dataset, rs)
        
    fn = "results/{}.csv".format(args.dataset)
    with open(fn, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(output)




