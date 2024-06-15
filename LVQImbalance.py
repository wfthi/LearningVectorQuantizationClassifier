"""
Data Augmentation for imbalanced classes using the LVQ algorithm

Use the LVQ algorithm to find a prototype of the classes in an imbalance
dataset. Then use the prototype to generate data of the minority class
(here binary date 0 or 1)

    Example of imbalanced datasets
    - Disease diagnosis.
    - Customer churn prediction.
    - Fraud detection.
    - Natural disaster and any rare natural events
    - Outliers

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


def init_lvq3(X, y, seed=2024):
    """
    Given the training set, compute the input weight/prototype for the LVQ3
    learnign method.

    Parameters
    ----------
    X : array-like
        training input

    y : array-like
        labels

    seeds : int
        random generator seed value

    Example
    -------
    >>> from LVQImbalance import *
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from LVQImbalance import *
    >>> X, y = make_classification(n_samples=10000, n_features=2,
    ...                            n_redundant=0,
    ...                            n_clusters_per_class=1,
    ...                            weights=[0.99], flip_y=0,
    ...                            random_state=1)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X)
    >>> Xs = scaler.transform(X)
    >>> Xm, ym, W = init_lvq3(Xs, y)
    >>> Xm, ym, W = init_lvq3(Xs, y, seed=0)
    """
    classes = np.unique(y)
    ly = len(y)
    ind = np.arange(0, ly, 1)
    train_idx = []
    rng = np.random.default_rng(seed)
    for c in classes:
        idx = rng.choice(ind[y == c], size=1, replace=False)
        train_idx.append(idx[0])
    print('Train indices:', train_idx)
    W = X[train_idx].astype(np.float64)
    mask = np.ones(ly, dtype=bool)
    mask[train_idx] = False
    return X[mask, :], y[mask], W


# Train using LVQ 3
# LVQ can be used to generate prototypes
def lvq3_train(X, y, W, a, b, max_ep, min_a, e, verbose=False):
    """
    Perform training/fitting using the LVQ3  model

    Parameters
    ----------
    X : array-like
        training input

    y : array-like
        labels

    a : float
        initial learning rate < 1, it should be around 0.1-0.5

    b : float
        b < 1
        learning rate decrease rate such that a_new = a_current * b

    W : array-like
        initial input prototype, see init_lvq3 for more details

    max_ep : int
        the maximum number of epochs

    min_a : float
        the minimum learning rate

    e : float
        dumping rate with 0 < e < 1

    verbose: bool, optional, default=False
        if True, the learning status will be displayed on the screen

    Return
    ------
    W : array-like
        the updated (learned) prototypes/weights

    c : 1D array
        the unique classes

    Example
    -------
    >>> Wout, c = lvq3_train(X, y, Winit, a, b, max_ep, min_a, e)
    """
    c = np.unique(y)
    r = c
    ep = 0

    while ep < max_ep:
        if verbose:
            print('Epoch:', ep + 1, '/', max_ep, 'learning rate:',
                  np.round(a, 5),
                  'min learning rate:', np.round(min_a, 5))
        for i, x in enumerate(X):
            d = [np.sum((w - x)**2) for w in W]
            min_1 = np.argmin(d)
            min_2 = 0
            dc = float(np.amin(d))
            dr = 0
            min_2 = d.index(sorted(d)[1])
            dr = float(d[min_2])
            if c[min_1] == y[i] and c[min_1] != r[min_2]:
                W[min_1] = W[min_1] + a * (x - W[min_1])

            elif c[min_1] != r[min_2] and y[i] == r[min_2]:
                if dc != 0 and dr != 0:
                    if min((dc / dr), (dr / dc)) > (1. - e) / (1. + e):
                        W[min_1] = W[min_1] - a * (x - W[min_1])
                        W[min_2] = W[min_2] + a * (x - W[min_2])
            elif c[min_1] == r[min_2] and y[i] == r[min_2]:
                W[min_1] = W[min_1] + e * a * (x - W[min_1])
                W[min_2] = W[min_2] + e * a * (x - W[min_2])
        a = a * b
        a = np.max([a, min_a])
        ep += 1
    return W, c


# Test Using LVQ 3
def lvq3_predict_proba(X, W, nb_importance):
    """
    Make predictions (probavility)

    Parameters
    ----------
    X : array-like
        test set in the same format than the training set

    W : array-like
        trained prototype

    nb_importance : int
        the number of entries in the input to be used to measure the 
        distance from a test input to the prototype, usually = the length of
        W

    Return
    ------
        :float
        the probabilities
    """
    W1, _ = W
    diff = np.abs(W1[1] - W1[0])
    j = np.flip(np.argsort(diff))
    j = j[0: nb_importance]
    d = np.array([np.sum(((w[j] - X[:, j])**2), 1) for w in W1])
    return d / d.sum(0)


def train_lvq(X, y, random_seed=2024, number_epochs=100, verbose=False):
    """
    Set the input required input and train a LVQ3 model with standard default
    values

    Example
    -------
    >>> from LVQImbalance import *
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> import matplotlib.pyplot as plt
    >>> from LVQImbalance import *
    >>> X, y = make_classification(n_samples=10000, n_features=2,
    ...                            n_redundant=0,
    ...                            n_clusters_per_class=1,
    ...                            weights=[0.99], flip_y=0,
    ...                            random_state=1)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X)
    >>> Xs = scaler.transform(X)
    >>> # train from 50 splits
    >>> skf = StratifiedKFold(n_splits=50)
    >>> epochs = 2
    >>> W0, W1 = [], []
    >>> for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    ...     WW = train_lvq(Xs[train_index], y[train_index],
    ...                    random_seed=i, number_epochs=epochs)
    ...     W = WW[0]
    ...     W0.append(W[0])
    ...     W1.append(W[1])
    >>> W0 = np.array(W0)
    >>> W1 = np.array(W1)
    >>> plt.scatter(Xs[y==0, 0], Xs[y==0, 1], s=3, label='0')
    >>> plt.scatter(Xs[y==1, 0], Xs[y==1, 1], s=3, label='1')
    >>> plt.scatter(W0[:,0], W0[:,1], s=5)
    >>> plt.scatter(W1[:,0], W1[:,1], s=5)
    >>> plt.show()
    """
    learning_starting_rate = 0.05  # 0.2
    learning_rate_decrease = 0.95
    minimum_learning_rate = 0.001
    m = 0.3  # 0.1 < m < 0.5 is a stabilizing constant.
    X_lvq_train, y_lvq_train, W_lvq = init_lvq3(X, y, seed=random_seed)
    W = lvq3_train(X_lvq_train, y_lvq_train, W_lvq, learning_starting_rate,
                   learning_rate_decrease,
                   number_epochs, minimum_learning_rate, m, verbose=verbose)
    return W


def lvq_extra(X, y, W, hot_encoding=False):
    """
    Balance the classes from an imbalance input set
    (e.g. input with outliers).

    The routine will use a LVQ3-trained prototype to generate randomly
    extra data for the training. The input are binary (one-hot encoding)
    0 or 1

    The limitation is that this is an augmentation based on a
    single prototype

    Parameters
    ----------
    X : array-like
        original input data set scaled to values beween 0 and 1

    y : array-like
        original input classes

    W : array-like
        LVQ-trained prototype for the minority class

    hot_encoding : boolean, optional, default=False
        True if all the features are only 0 or 1 (hot-encoding)

    Return
    ------
    X_extra : array-like with the same rank than X
        balanced set with augmented data, the features should have
        values between 0 and 1

    y_extra : array-like
        balanced classes

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from LVQImbalance import *
    >>> X, y = make_classification(n_samples=10000, n_features=2,
    ...                            n_redundant=0,
    ...                            n_clusters_per_class=1,
    ...                            weights=[0.99], flip_y=0,
    ...                            random_state=1)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X)
    >>> Xs = scaler.transform(X)
    >>> plt.scatter(Xs[y==0, 0], Xs[y==0, 1], s=3, label='0')
    >>> plt.scatter(Xs[y==1, 0], Xs[y==1, 1], s=3, label='1')
    >>> plt.legend()
    >>> plt.show()
    >>> Counter(y)
    >>> W = train_lvq(Xs, y, verbose=True, number_epochs=30)
    >>> X_extra, y_extra = lvq_extra(Xs, y, W)
    >>> Counter(y_extra)
    >>> w_extra = y_extra == 0
    >>> plt.scatter(X_extra[w_extra, 0], X_extra[w_extra, 1], s=1,
    ...             label='0', alpha=0.5)
    >>> plt.scatter(X_extra[~w_extra, 0], X_extra[~w_extra, 1], s=1,
    ...             label='1', alpha=0.5)
    >>> plt.scatter(Xs[y==0, 0], Xs[y==0, 1], s=3, label='0', alpha=0.9)
    >>> plt.scatter(Xs[y==1, 0], Xs[y==1, 1], s=3, label='1',
    ...             alpha=0.9, color='black')
    >>> plt.legend()
    >>> plt.show()
    >>> # Test
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> # define model
    >>> model = DecisionTreeClassifier()
    >>> # evaluate
    >>> cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    >>> scores = cross_val_score(model, X_extra, y_extra, scoring='roc_auc',
    ...                          cv=cv, n_jobs=-1)
    >>> print('Mean ROC AUC: %.3f' % np.mean(scores))
    >>> #  Mean ROC AUC: 0.990  # 30 epochs
    """
    if X.min() < 0:
        print("There are features with value < 0")
        return None, None
    if X.max() > 1.:
        print("There are features with value > 1")
        return None, None
    W0 = W[0]
    trh = W0[1]
    count_train = Counter(y)
    nb_extra = (count_train[0] - count_train[1])
    if nb_extra < 0:
        minority = 0
        nb_extra = -nb_extra
    else:
        minority = 1
    X_pos_extra = np.empty((nb_extra, X.shape[1]))
    if hot_encoding:
        rnd = np.random.random((nb_extra, X.shape[1]))
        X_pos_extra = (rnd < trh) * 1  # augmentated value 0 or 1
    else:
        imin, imax = 0, 0
        while imax < nb_extra:
            rnd = np.random.random((nb_extra, X.shape[1]))
            dist = np.array([np.sum((rnd - WW)**2, 1) for WW in W0])
            w_new = np.argsort(dist, 0)[0] == minority
            nb_new = np.count_nonzero(w_new)
            nb_more = np.min([nb_extra - imax, nb_new])
            imin = imax
            imax += nb_more
            X_pos_extra[imin:imax, :] = rnd[w_new, :][0:nb_more]

    X_extra = np.vstack((X_pos_extra, X))
    y_extra = np.append(np.full(X_pos_extra.shape[0], 1), y)
    return X_extra, y_extra


def lvq_prototypes(n_prototypes, X, y, number_epochs=10, seed=1):
    """
    Balance the binary classes with multiple prototypes.

    The code will split the input into n_prototypes separate batches.
    Then it will train for each batch data for number_epochs for determine
    the prototypes.

    The code then will run the extra data routine lvq_extra for each prototype
    pairs to balance the classes.

    Parameters
    ----------
    n_prototypes : int
        the number of prototype per class.
        n_prototypes = 2 means 2 for each class in y

    X : array-like
        original input data set scaled to values beween 0 and 1

    y : array-like
        original input classes

    number_epochs : int, optional, default = 10
        the number of epochs used for the lvq training

    seed : int, optional, default = 1
        random generator seed value

    Returns
    -------
    Xel : list
        list of original + extra datasets
        one entry per prototype

    yel : list
        list of original + extra corresponding classes

    W0: list
        the weights for the class 0

    W1 : list
        the weights for the class 1

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from LVQImbalance import *
    >>> X, y = make_classification(n_samples=10000, n_features=2,
    ...                            n_redundant=0,
    ...                            n_clusters_per_class=1,
    ...                            weights=[0.99], flip_y=0,
    ...                            random_state=1)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X)
    >>> Xs = scaler.transform(X)
    >>> X_extra, y_extra, Xel, yel, W0, W1 = lvq_prototypes(10, Xs, y)
    >>> for i, (Xe, ye) in enumerate(zip(Xel, yel)):
    ...     pos = ye == 0
    ...     plt.scatter(Xe[pos, 0], Xe[pos, 1], s=1, 
    ...                 label='extra 0', alpha=0.5)
    ...     plt.scatter(Xe[~pos, 0], Xe[~pos, 1], s=1, 
    ...                 label='extra 1', alpha=0.5)
    >>> plt.scatter(Xs[y==0, 0], Xs[y==0, 1], s=3, alpha=0.5,
    ...             label='Original 0')
    >>> plt.scatter(Xs[y==1, 0], Xs[y==1, 1], s=3, label='Original 1',
    ...             alpha=0.5, color='black')
    >>> plt.scatter(W0[0], W0[1], label='Weight 0', marker='*')
    >>> plt.scatter(W1[0], W1[1], label='Weight 1', marker='*')
    >>> plt.legend()
    >>> plt.show()
    >>> # Test
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> # define model
    >>> model = DecisionTreeClassifier()
    >>> # evaluate
    >>> cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    >>> scores = cross_val_score(model, X_extra, y_extra, scoring='roc_auc',
    ...                          cv=cv, n_jobs=-1)
    >>> print('Mean ROC AUC: %.3f' % np.mean(scores))
    >>> # Mean ROC AUC: 0.990 with 5 prototypes, number_epochs = 10
    """
    uX = np.unique(X)
    hot_encoding = False
    # check if the features are all hot-encodings
    if len(uX) == 2:
        if all(np.unique(X) == [0., 1.]):
            hot_encoding = True
    ind_list = []
    split = [int(X.shape[0] / n_prototypes)] * (n_prototypes - 1)
    split.append(X.shape[0] - sum(split))
    lind = X.shape[0]
    indall = np.arange(0, lind, 1)
    yall = y.copy()
    count = Counter(y)
    ly = len(y)
    rng = np.random.default_rng(seed)
    for i, sp in enumerate(split):
        ind = np.arange(0, lind, 1)
        w0 = np.where(yall == 0)[0]
        w1 = np.where(yall == 1)[0]
        mask = np.ones(len(ind), dtype=bool)
        if i < n_prototypes - 1:  # stratifield choice
            sp1 = int(np.max([1, sp * count[1] / ly]))
            sp0 = sp - sp1
            mask[rng.choice(ind[w0], sp0, replace=False)] = False
            if len(w1) == 0:
                print('Not enough entry for class 1')
                print('Try decreasing the nunber of prototypes')
                return None, None, None, None, None, None
            mask[rng.choice(ind[w1], sp1, replace=False)] = False
        else:  # last batch, use the rest of the data
            mask[rng.choice(ind, sp, replace=False)] = False
        ind_list.append(indall[~mask])
        indall = indall[mask]
        yall = yall[mask]
        lind -= sp
    W0, W1, Xel, yel = [], [], [], []
    for i, ind in enumerate(ind_list):
        W = train_lvq(X[ind], y[ind],
                      verbose=True, number_epochs=number_epochs)
        W0.append(W[0][0])
        W1.append(W[0][1])
        Xe, ye = lvq_extra(X[ind], y[ind], W, hot_encoding=hot_encoding)
        if i == 0:
            X_extra = Xe.copy()
            y_extra = ye.copy()
        X_extra = np.vstack((X_extra, Xe))
        y_extra = np.concatenate((y_extra, ye))
        Xel.append(Xe)
        yel.append(ye)
    W0 = np.array(W0).T
    W1 = np.array(W1).T
    return X_extra, y_extra, Xel, yel, W0, W1
