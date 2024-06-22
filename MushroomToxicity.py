"""
    Slightly imbalanced dataset binary classification

    Comparison between imbalanced data augmentation techniques such as
    SMOTE, LVQ, Tree-LVQ

    Mushroom data creation, curation, and simulation to support classification
    tasks
    By Dennis Wagner, D. Heider, Georges Hattab. 2021
    Published in Scientific Reports

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
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from LVQImbalance import tree_lvq, lvq_prototypes
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def cv_training(model, X_train, y_train, X_test, y_test):
    # Cross-validation training
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(model, X_train, y_train,
                            scoring='average_precision',
                            cv=cv, n_jobs=-1, return_estimator=True)
    print('Mean average precision training set: %.3f' %
          np.mean(scores['test_score']))
    proba = []
    for mod in scores['estimator']:
        proba.append(mod.predict_proba(X_test))
    mean_proba = np.mean(proba, 0)
    print('Average precision on the test set: %.3f' %
          average_precision_score(y_test, mean_proba[:, 1]))
    print('Cohen kappa on the test set: %.3f' %
          cohen_kappa_score(y_test, np.rint(mean_proba[:, 1])))
    print('Matthews correlation coefficient  on the test set: %.3f' %
          matthews_corrcoef(y_test, np.rint(mean_proba[:, 1])))
    return mean_proba


# fetch dataset
secondary_mushroom = fetch_ucirepo(id=848)
# metadata
print(secondary_mushroom.metadata)
# variable information
print(secondary_mushroom.variables)

# data (as pandas dataframes)
X = secondary_mushroom.data.features.values
y = secondary_mushroom.data.targets.values

# remove columns with missing data
wok = secondary_mushroom.variables.missing_values.values[1:] == 'no'
X = X[:, wok]

# convert categorial data into continuous values between 0 and 1
# instead of feature hot-encoding that increases the dimensionality
# especially if the number of categories is high
cat = secondary_mushroom.variables.type[1:][wok]
for i, ctype in enumerate(cat):
    if ctype == 'Categorical':
        lx = len(set(X[:, i])) - 1
        dict_featurers = {k: i / lx for i, k in enumerate(set(X[:, i]))}
        X[:, i] = [dict_featurers[f] for f in X[:, i]]
X = X.astype(float)
# Scale the features in the 0 to 1 range
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
Xs = scaler.transform(X)

# change to p -> 1, e -> 0
# edible=e and poisonous=p
w = y == 'p'
y[w] = 1  # class 1 toxic
y[~w] = 0
y = y.reshape(y.shape[0]).astype(int)
count = Counter(y)

# Now we split between training and test sets
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.1,
                                                    stratify=y,
                                                    random_state=42)

# Data augmentation with the SMOTE method
sm = SMOTE(sampling_strategy=1.0, random_state=1235)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Cross-validation training
model = RandomForestClassifier()
mean_proba = cv_training(model, X_res, y_res, X_test, y_test)

# plot the confusion matrix
cm = confusion_matrix(y_test, np.rint(mean_proba[:, 1]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Data augmentation with the LVQ
# minority class is 0 (non-toxic)
n_prototypes = 10

X_res, y_res, _, _, _, _ = lvq_prototypes(n_prototypes, X_train, y_train)

# Cross-validation training
model = RandomForestClassifier()
mean_proba = cv_training(model, X_res, y_res, X_test, y_test)

# plot the confusion matrix
cm = confusion_matrix(y_test, np.rint(mean_proba[:, 1]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# tree-LVQ
count = Counter(y_train)
nb_extra = count[1] - count[0] - n_prototypes
X_lvq, y_lvq, W0, _ = tree_lvq(n_prototypes, X_train, y_train,
                               kneighbours=6000, iter_max_fac=1000,
                               append=False,
                               number_epochs=30,
                               verbose=True,
                               nb_extra=nb_extra)
X_res = np.vstack((X_train, X_lvq, W0))
y_res = np.concatenate((y_train, y_lvq, np.full(n_prototypes, 0)))

# Cross-validation training
model = RandomForestClassifier()
mean_proba = cv_training(model, X_res, y_res, X_test, y_test)

# plot the confusion matrix
cm = confusion_matrix(y_test, np.rint(mean_proba[:, 1]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
