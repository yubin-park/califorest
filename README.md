# CaliForests

**Cali**brated Random **Forests**

This Python package implements the CaliForest algorithm presented in [ACM CHIL 2020](https://www.chilconference.org/).

## Installing

Installing from the source:

```
$ git clone git@github.com:yubin-park/califorest.git
$ cd califorest
$ python setup.py develop
```

## Example Code

```
from califorest import CaliForest

model = CaliForest(n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=mss,
                    min_samples_leaf=msl,
                    ctype="isotonic")

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:,1]
```

## Reference

Y. Park and J. C. Ho. 2020. **CaliForest: Calibrated Random Forest for Health Data**. *ACM Conference on Health, Inference, and Learning (2020)*




