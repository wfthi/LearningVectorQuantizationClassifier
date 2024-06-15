"""
    Examples of imblanced dataset rebalanced wuthe the LVQ method

    The data are taken from the imbalanced-learn list of datasets

    Author: Wing-Fai Thi <wingfai.thi googlemail. com>

    License: GNU v3.0

    Copyright (C) 2024  Wing-Fai Thi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Python : > version 3

"""
from collections import Counter
import numpy as np
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from LVQImbalance import lvq_prototypes

# Example 1: Solar Flare prediction
solar_flare = fetch_datasets()['solar_flare_m0']
solar_flare.data.shape
count = Counter(solar_flare.target)
X = solar_flare.data
y = solar_flare.target
y[y == -1] = 0
X_extra, y_extra, Xel, yel, W0, W1 = lvq_prototypes(3, X, y)
model = DecisionTreeClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_validate(model, X_extra, y_extra, scoring='average_precision',
                        cv=cv, n_jobs=-1, return_estimator=True)
print('Mean average precision training set: %.3f'
      % np.mean(scores['test_score']))
# Mean average precision on the CV training set

# Now we split between training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42)
X_extra, y_extra, Xel, yel, W0, W1 = lvq_prototypes(5, X_train, y_train,
                                                    number_epochs=30)
model = RandomForestClassifier()
# Repeated cross-validation training
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_validate(model, X_extra, y_extra, scoring='average_precision',
                        cv=cv, n_jobs=-1, return_estimator=True)
print('Mean average precision training set: %.3f' %
      np.mean(scores['test_score']))
proba = []
for mod in scores['estimator']:
    proba.append(mod.predict_proba(X_test))
mean_proba = np.mean(proba, 0)
print('Average precision on the test set: %.3f' %
      average_precision_score(y_test, mean_proba[:, 1]))

# Example 2 protein homology prediction problem
protein = fetch_datasets()['protein_homo']
Xp = protein.data
yp = protein.target
yp[yp == -1] = 0
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(Xp)
Xs = scaler.transform(Xp)
Xs[Xs > 1.0] = 1.0  # Minmax scaler may show numerical precision error
X_train, X_test, y_train, y_test = train_test_split(Xs, yp, test_size=0.2,
                                                    random_state=42)
Xp_extra, yp_extra, Xpel, ypel, Wp0, Wp1 = lvq_prototypes(20, X_train, y_train,
                                                          number_epochs=30)
model = DecisionTreeClassifier()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
scores = cross_validate(model, Xp_extra, yp_extra, scoring='average_precision',
                        cv=cv, n_jobs=-1, return_estimator=True)
print('Mean average precision training set: %.3f'
      % np.mean(scores['test_score']))
proba = []
for mod in scores['estimator']:
    proba.append(mod.predict_proba(X_test))
mean_proba = np.mean(proba, 0)
print('Average precision on the test set: %.3f' %
      average_precision_score(y_test, mean_proba[:, 1]))
# Average precision on the test set: 0.794 with DecisionTree

# Use XGBoost classifier with early stopping
X_tr, X_eval, y_tr, y_eval = train_test_split(Xp_extra, yp_extra,
                                              stratify=yp_extra,
                                              random_state=94)

# Use "hist" for constructing the trees, with early stopping enabled.
clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=5)
# Fit the model, test sets are used for early stopping.
clf.fit(X_tr, y_tr, eval_set=[(X_eval, y_eval)])
proba_xgb = clf.predict_proba(X_test)
print('XGBoost Average precision on the test set: %.3f' %
      average_precision_score(y_test, proba_xgb[:, 1]))

# plot the confusion matrix
cm = confusion_matrix(y_test, np.rint(proba_xgb[:, 1]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
