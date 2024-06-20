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
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans


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
def lvq3_train(X, y, Winit, a, b, max_ep, min_a, e, verbose=False):
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

    Winit : array-like
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
    W = Winit.copy()

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


def lvq_extra(X, y, W, sampling_strategy=1., direct_hot_encoding=False,
              seed=1, nb_extra=None, append=True,
              hot_encoding=False, verbose=False, data_boundary=True):
    """
    Balance the classes from an imbalance input set
    (e.g. input with outliers).

    The routine will use a LVQ3-trained prototype to generate randomly
    extra data for the training. The input are binary (one-hot encoding)
    0 or 1

    The limitation is that this is an augmentation based on a
    single prototype. The data_boundary = True is needed if one
    wishes to to asymptotically regenerate the distribution of the
    minority class.

    Parameters
    ----------
    X : array-like
        original input data set scaled to values beween 0 and 1

    y : array-like
        original input classes

    W : array-like
        LVQ-trained prototype for the minority class

    nb_extra : int, optional, default=None
        the request number of symthetic data
        if None, the code will use the imbalance ratio and the
        sampling_strategy value.

    append : boolean, optional, default=True
        append the augmented data to the input

    seed: int, optional, default=1
        random generator seed

    sampling_strategy : float, optional, default=1.0
        the fraction between the two classes.
        sampling_strategy = 1. means that both classes will be balanced

    direct_hot_encoding : boolean, optional, default=False
        Use an alternative method to augment hot-encoding data
        w1 is a value the prototype W1 for class 1
        if random value(0, 1) < w1 : value = 1 else value = 0
        For example if w1 = 0.2, there is 20% chance to get a value of 1
        and 80% chance to get a value of 0

    hot_encoding : boolean, optional, default=False
        True if all the features are only 0 or 1 (hot-encoding)

    verbose : boolean, optional, default=False
        True to have screen logging

    data_boundary : boolean, optional, default=True
        if True, bound the augmented data within the range of the input
        data, i.e. the newly generated dataset is within the convex hull
        of the minority train data.
        It may be worth to extend beyond the boundary values to
        make the learning more general.

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
    if sampling_strategy == 0.:
        return X, y
    if (sampling_strategy > 1.) or (sampling_strategy < 0.):
        print("sampling_strategy has to be in [0, 1]")
        return None, None
    if X.min() < 0:
        print("There are features with value < 0")
        return None, None
    if X.max() > 1.:
        print("There are features with value > 1")
        return None, None
    count_train = Counter(y)
    if count_train[1] > count_train[0]:
        majority = 1
        minority = 0
    else:
        majority = 0
        minority = 1
    if nb_extra is None:
        nb_extra = int(np.ceil(sampling_strategy * count_train[majority] -
                               count_train[minority]))
    if verbose:
        print('Require', nb_extra, 'data')
    if verbose:
        print('Minority class', minority)
    wpos = y == minority
    Xmin = X[wpos, :].min(0)
    Xmax = X[wpos, :].max(0)
    X_pos_extra = np.empty((nb_extra, X.shape[1]))
    count = Counter(y)
    if verbose:
        print('Majority:', count[majority], 'Minority:', count[minority])
    ratio = int(count[majority] / count[minority])
    max_fail = 100
    fail = 0
    imin, imax = 0, 0
    ratio_fac = 10
    if verbose:
        print('Using seed:', seed)
    rng = np.random.default_rng(seed)
    if direct_hot_encoding:
        if verbose:
            print('Hot encoding')
        rnd = rng.random((nb_extra, X.shape[1]))
        if data_boundary:
            rnd = rnd * (Xmax - Xmin) + Xmin
        X_pos_extra = (rnd < W[0][1]) * 1  # augmented value 0 or 1
        if append:
            X_extra = np.vstack((X_pos_extra, X))
            y_extra = np.append(np.full(X_pos_extra.shape[0], 1), y)
        else:
            X_extra = X_pos_extra
            y_extra = np.full(X_pos_extra.shape[0], 1)
        return X_extra, y_extra

    while imax < nb_extra:
        nb_sample = int(ratio * nb_extra)
        if verbose:
            print('nb_extra:', nb_extra, 'ratio:', ratio,
                  'nb_sample:', nb_sample)
        rnd = rng.random((nb_sample, X.shape[1]))
        if data_boundary:
            rnd = rnd * (Xmax - Xmin) + Xmin
        if hot_encoding:
            rnd = np.rint(rnd).astype(int)
        dist = np.array([np.sum((rnd - WW)**2, 1) for WW in W[0]])
        # only criterion is that the distance is closer to the minority
        # prototype
        w_new = np.argsort(dist, 0)[0] == minority
        nb_new = np.count_nonzero(w_new)
        if verbose:
            print('nb_new:', nb_new)
        if nb_new == 0:
            if verbose:
                print('No new data from all the random draws')
            ratio = ratio * ratio_fac
            fail += 1
            if fail > max_fail:
                print('Augmentation failed')
                return None, None
        ratio = np.min([ratio, 1000])
        nb_more = np.min([nb_extra - imax, nb_new])
        imin = imax
        imax += nb_more
        X_pos_extra[imin:imax, :] = rnd[w_new, :][0:nb_more]
    # Check
    d0 = np.sum((X_pos_extra - W[0][0])**2, 1)
    d1 = np.sum((X_pos_extra - W[0][1])**2, 1)
    assert np.count_nonzero((d1 < d0)) == X_pos_extra.shape[0]
    d0 = np.sum((X - W[0][0])**2, 1)
    d1 = np.sum((X - W[0][1])**2, 1)
    if append:
        X_extra = np.vstack((X_pos_extra, X))
        y_extra = np.append(np.full(X_pos_extra.shape[0], 1), y)
    else:
        X_extra = X_pos_extra
        y_extra = np.full(X_pos_extra.shape[0], 1)
    return X_extra, y_extra


def random_split(n_prototypes, X, y, seed=1):
    """
    Split the input into n_prototypes batches. It returns a list of
    indices for each batch

    Parameters
    ----------
    n_prototypes : int
        the number of prototype per class.
        n_prototypes = 2 means 2 for each class in y

    X : array-like
        original input data set scaled to values beween 0 and 1

    y : array-like
        original input classes

    seed : int, optional, default = 1
        random generator seed value

    Return
    ------
     : list of arrays
        a list of indices for each batch (n_prototypes batches)
    """
    ind_list = []
    split = [int(X.shape[0] / n_prototypes)] * (n_prototypes - 1)
    split.append(X.shape[0] - sum(split))
    lind = X.shape[0]
    indall = np.arange(0, lind, 1)
    yall = y.copy()
    count = Counter(y)
    ly = len(y)
    rng = np.random.default_rng(seed)
    print('Split randomly the input dataset into batches')
    for i, sp in enumerate(split):
        print('Batch ', i + 1, '/', n_prototypes)
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
    return ind_list


def lvq_prototypes(n_prototypes, X, y, number_epochs=10,
                   sampling_strategy=1., direct_hot_encoding=False,
                   hot_encoding=False, append=True,
                   seed=1, verbose=False, data_boundary=True):
    """
    Balance the binary classes with multiple prototypes.

    The code will split the input into n_prototypes separate batches.
    Then it will train for each batch data for number_epochs for determine
    the prototypes.

    The code then will run the extra data routine lvq_extra for each prototype
    pairs to balance the classes.

    The generated data will be within the convex hull of the minority data
    if the flag data_boundary is set to True.

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

    append : boolean, optional, default=True
        append the augmented data to the input

    hot_encoding : boolean, optional, default = False
        whether the features should be 0 or 1 only

    direct_hot_encoding : boolean, optional, default = False
        Use an alternative method to augment hot-encoding data

    sampling_strategy : float, optional, default=1.0
        the fraction between the two classes.
        sampling_strategy = 1. means that both classes will be balanced

    verbose : boolean, optional, default=False
        True to have screen logging

    data_boundary : boolean, optional, default=True
        if True, bound the augmented data within the range of the input

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
    >>> from imblearn.over_sampling import SMOTE
    >>> from LVQImbalance import *
    >>> X, y = make_classification(n_samples=10000, n_features=2,
    ...                            n_redundant=0,
    ...                            n_clusters_per_class=1,
    ...                            weights=[0.99], flip_y=0,
    ...                            random_state=1)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X)
    >>> Xs = scaler.transform(X)
    >>> X_extra, y_extra, Xel, yel, W0, W1 = lvq_prototypes(10, Xs, y,
    ...                                                     data_boundary=True)
    >>> # SMOTE
    >>> sm = SMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(Xs, y)
    >>> for i, (Xe, ye) in enumerate(zip(Xel, yel)):
    ...     pos = ye == 0
    ...     if i == 0:
    ...         plt.scatter(Xe[~pos, 0], Xe[~pos, 1], s=5,
    ...                    label='LVQ', color='red', alpha=0.5)
    ...     else:
    ...         plt.scatter(Xe[~pos, 0], Xe[~pos, 1], s=5,
    ...                     color='red', alpha=0.5)
    >>> plt.scatter(Xs[y==0, 0], Xs[y==0, 1], s=5, alpha=0.5,
    ...             label='Original 0', color='blue')
    >>> plt.scatter(Xs[y==1, 0], Xs[y==1, 1], s=5, label='Original 1',
    ...             alpha=0.5, color='black')
    >>> plt.scatter(W0[0], W0[1], label='Weight 0', marker='*')
    >>> plt.scatter(W1[0], W1[1], label='Weight 1', marker='*')
    >>> plt.xlabel('Feature 1')
    >>> plt.ylabel('Feature 2')
    >>> plt.legend()
    >>> plt.show()
    >>> # Plot SMOTE resampling
    >>> plt.scatter(X_res[y_res ==1, 0], X_res[y_res ==1, 1],
    ...             label='SMOTE 1', color='red', marker='*', s=1)
    >>> plt.scatter(Xs[y==0, 0], Xs[y==0, 1], s=5, alpha=0.5,
    ...             label='Original 0')
    >>> plt.scatter(Xs[y==1, 0], Xs[y==1, 1], s=5, label='Original 1',
    ...             alpha=0.5, color='black')
    >>> plt.xlabel('Feature 1')
    >>> plt.ylabel('Feature 2')
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
    >>> # Mean ROC AUC: 0.990 with 5-10 prototypes, number_epochs = 10
    """
    # check if the features are all hot-encodings
    uX = np.unique(X)
    mssg = "Warning: all features are 0 and 1 but the augmented will be floats"
    if len(uX) == 2:
        if all(np.unique(X) == [0., 1.]):
            if not hot_encoding:
                print(mssg)
                print('set hot_encoding=True to have only 0 or 1')
    ind_list = random_split(n_prototypes, X, y, seed=seed)
    W0, W1, Xel, yel = [], [], [], []
    for i, ind in enumerate(ind_list):
        print('Training batch ', i + 1, '/', n_prototypes)
        W = train_lvq(X[ind], y[ind], random_seed=i,
                      verbose=True, number_epochs=number_epochs)
        ypred = lvq3_predict_proba(X[ind], W, len(W))
        print('LVQ training average precision:',
              average_precision_score(y[ind], ypred[1, :]))
        print('LVQ training Cohen kappa:',
              cohen_kappa_score(y[ind], np.rint(ypred[1, :])))
        W0.append(W[0][0])  # class 0
        W1.append(W[0][1])
        print('... augment the data')
        if data_boundary:
            print('data_boundary:', data_boundary)
        Xe, ye = lvq_extra(X[ind], y[ind], W, verbose=verbose,
                           seed=seed,
                           append=append,
                           sampling_strategy=sampling_strategy,
                           direct_hot_encoding=direct_hot_encoding,
                           data_boundary=data_boundary,
                           hot_encoding=hot_encoding)
        if Xe is None and ye is None:
            return None, None, None, None, W0, W1
        if i == 0:
            X_extra = Xe.copy()
            y_extra = ye.copy()
        else:
            X_extra = np.vstack((X_extra, Xe))
            y_extra = np.concatenate((y_extra, ye))
        Xel.append(Xe)
        yel.append(ye)
    W0 = np.array(W0).T
    W1 = np.array(W1).T
    return X_extra, y_extra, Xel, yel, W0, W1


def tree_lvq(n_prototypes, X, y, nb_extra=None, seed=1,
             append=True,
             kneighbours=30, number_epochs=10,
             data_boundary=True,
             sampling_strategy=1.0,
             learning_starting_rate=0.05,
             learning_rate_decrease=0.95,
             minimum_learning_rate=0.001,
             m=0.3,
             iter_max_fac=10,
             verbose=False):
    """
    Local LVQ learning by splitting the data. kneighbours is found
    around (distance^2 metric) a randomly chosen minority instance and LVQ
    is used to find a local prototype for each class. n_prototypes
    prototype-pairs are created. To syntheize new data, random values are
    computed and the closest distance prototype-pair is chosen and the
    distance between the two classes is used to accept the the new values as
    new data.
    The strategy is unlike lvq_prototypes where the splitting is done by
    drawing randomly a sample within the entire dataset. This method can
    result in prototypes that cover more uniformely the feature space.

    The application of SMOTE (0.5-0.8 sampling strategy) + LVQ is very powerful
    as the two techniques complement well each other.

    See also Bordeline-SMOTE

    Parameters
    ----------
    n_prototypes : int
        the number of prototype per class.
        n_prototypes = 2 means 2 for each class in y

    X : array-like
        original input data set scaled to values beween 0 and 1

    y : array-like
        original input classes

    kneighbours : int, optional, default=30
        the number of neighbouring points to use for the local learning

    number_epochs : int, optional, default = 10
        the number of epochs used for the lvq training

    seed : int, optional, default = 1
        random generator seed value

    append : boolean, optional, default=True
        append the augmented data to the input

    nb_extra : int, optional, default=None
        the number of extra minority data point to be generated
        if None, the diffference in the input between the two
        classes is used together with the value of sampling_strategy

    sampling_strategy : float, optional, default=1.0
        the fraction between the two classes.
        sampling_strategy = 1. means that both classes will be balanced

    verbose : boolean, optional, default=False
        True to have screen logging

    data_boundary : boolean, optional, default=True
        if True, bound the augmented data within the range of the input

    learning_starting_rate : float, optional, default=0.05
        the starting value of the learning rate for the LVQ3

    learning_rate_decrease : float, optional, default=0.95
        the learning rate decrease factor (< 1)

    minimum_learning_rate : float, optional, default=0.001
        the minimum learning rate

    m : float, optional, default=0.3
        a damping factor for the learning between 0 < m < 1

    iter_max_fac : optinal, int, default=10
        the number of attempts to find a new valid data per required
        new data

    Returns
    -------
    X_extra : list
        list of original + extra datasets
        one entry per prototype

    y_extra : list
        list of original + extra corresponding classes

    W0: list
        the weights for the class 0

    W1 : list
        the weights for the class 1

    Example
    -------
    >>> import numpy as np
    >>> from collections import Counter
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_moons, make_circles
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from imblearn.datasets import make_imbalance
    >>> from imblearn.over_sampling import SMOTE
    >>> from LVQImbalance import tree_lvq
    >>> # create the Moon sample
    >>> Xmoon, ymoon = make_moons(n_samples=600, noise=0.2, random_state=0)
    >>> sampling_strategy = {1: 50}
    >>> X_imb, y_imb = make_imbalance(Xmoon, ymoon,
    ...                               sampling_strategy=sampling_strategy)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X_imb)
    >>> Xs = scaler.transform(X_imb)
    >>> n_prototypes = 10
    >>> count = Counter(y_imb)
    >>> nb_extra = count[0] - count[1] - n_prototypes
    >>> X_lvq, y_lvq, W0, W1 = tree_lvq(n_prototypes, Xs, y_imb,
    ...                                 nb_extra=nb_extra)
    >>> X_extra = np.vstack((X_lvq, W1))
    >>> y_extra = np.append(y_lvq, np.full(n_prototypes, 1))
    >>> w = y_imb == 0
    >>> size = 10
    >>> plt.scatter(X_lvq[:, 0], X_lvq[:, 1], s=20, label='LVQ')
    >>> plt.scatter(Xs[w, 0], Xs[w, 1], s=size, label='Majority')
    >>> plt.scatter(Xs[~w, 0], Xs[~w, 1], s=size, label='Minority')
    >>> plt.scatter(W0[:, 0], W0[:, 1], s=50, marker='*', label='0')
    >>> plt.scatter(W1[:, 0], W1[:, 1], s=50, marker='*', label='1')
    >>> plt.xlabel('X1')
    >>> plt.ylabel('X2')
    >>> plt.legend()
    >>> plt.tight_layout()
    >>> plt.show()
    >>> # SMOTE + Tree-LVQ
    >>> sm = SMOTE(sampling_strategy=0.5, random_state=1235)
    >>> X_res, y_res = sm.fit_resample(Xs, y_imb)
    >>> n_prototypes = 10
    >>> count = Counter(y_res)
    >>> nb_extra = count[0] - count[1] - n_prototypes
    >>> X_lvq, y_lvq, W0, W1 = tree_lvq(n_prototypes, X_res, y_res,
    ...                                 append=False,
    ...                                 nb_extra=nb_extra, verbose=True)
    >>> X_extra = np.vstack((X_res, X_lvq, W1))
    >>> y_extra = np.concatenate((y_res, y_lvq, np.full(n_prototypes, 1)))
    >>> w = y_imb == 0
    >>> size = 10
    >>> plt.scatter(X_res[:, 0], X_res[:, 1], s=20, label='SMOTE')
    >>> plt.scatter(X_lvq[:, 0], X_lvq[:, 1], s=20, label='LVQ')
    >>> plt.scatter(Xs[w, 0], Xs[w, 1], s=size, label='Majority')
    >>> plt.scatter(Xs[~w, 0], Xs[~w, 1], s=size, label='Minority')
    >>> plt.scatter(W0[:, 0], W0[:, 1], s=50, marker='*', label='0')
    >>> plt.scatter(W1[:, 0], W1[:, 1], s=50, marker='*', label='1')
    >>> plt.xlabel('X1')
    >>> plt.ylabel('X2')
    >>> plt.legend()
    >>> plt.tight_layout()
    >>> plt.show()
    """
    assert X.shape[0] > kneighbours
    count = Counter(y)
    tree = BallTree(X, leaf_size=2)
    if count[0] > count[1]:
        majority = 0
        minority = 1
    else:
        majority = 1
        minority = 0
    wminority = np.where(y == minority)[0]
    W0, W1, W0init, W1init = [], [], [], []
    itry_max = 100
    for i in range(n_prototypes):
        cy = 0
        if verbose:
            print('Prototype:', i)
        itry = 0
        lmaj = 1
        while cy <= 1:
            itry += 1
            iminority = np.random.choice(wminority, replace=False)
            indtree = tree.query(X[iminority, :].reshape(1, -1),
                                 k=kneighbours,
                                 return_distance=False)
            wmajority = np.where(y[indtree[0][1:]] == majority)[0]
            lmaj = len(wmajority)
            if verbose:
                print('Number of majority sample', lmaj)
            if lmaj < 2:  # both the majority and minority sample >= 2
                continue
            # Find a majority instance among the neighbours
            imajority = indtree[0][1 + np.random.choice(wmajority)]
            Winit = np.stack((X[imajority, :], X[iminority, :]))
            W0init.append(Winit[0])
            W1init.append(Winit[1])
            ly = len(y)
            mask = np.zeros(ly, dtype=bool)
            mask[indtree[0][1:]] = True
            mask[iminority] = False
            mask[imajority] = False
            X_train = X[mask, :]
            y_train = y[mask]
            # 0.1 < m < 0.5 is a stabilizing constant.
            counter_y = Counter(y_train)
            cy = counter_y[minority]
            if itry > itry_max:
                print("Cannot find the vlid split")
                print("Try a lower number of prototypes or more neighbours")
                return None, None, None, None
        wminority = wminority[wminority != iminority]
        if verbose:
            print("Number of minority sample",  cy)
        W = lvq3_train(X_train, y_train, Winit,
                       learning_starting_rate,
                       learning_rate_decrease,
                       number_epochs,
                       minimum_learning_rate,
                       m,
                       verbose=verbose)
        W0.append(W[0][0])
        W1.append(W[0][1])
    W0 = np.array(W0)
    W1 = np.array(W1)
    # synthetic data generation
    if nb_extra is None:
        nb_extra = int(np.ceil(sampling_strategy * count[majority] -
                               count[minority]))
    wpos = y == minority
    Xmin = X[wpos, :].min(0)
    Xmax = X[wpos, :].max(0)
    iter = 0
    npos = 0
    iter_max = iter_max_fac * nb_extra
    pos = []
    nrnd = X.shape[1]
    while npos < nb_extra and iter < iter_max:
        iter += 1
        rng = np.random.default_rng(seed + iter)
        rnd = rng.random(nrnd)
        if data_boundary:
            rnd = rnd * (Xmax - Xmin) + Xmin
        d0 = np.sum((rnd - W0)**2, 1)
        d1 = np.sum((rnd - W1)**2, 1)
        # find the closest prototype pairs and compare
        # the distances to them
        imin = np.argmin(d0 + d1)
        if np.mean((d1[imin] < d0[imin]) * 1) > 0.5:
            pos.append(rnd)
            npos += 1
    if iter >= iter_max:
        print("Not enough synthetic data generated.")
        print("Please increase iter_max_fac")
    pos = np.array(pos)
    if append:
        X_extra = np.vstack((X, pos))
        y_extra = np.append(y, np.full(nb_extra, 1))
    else:
        X_extra = pos
        y_extra = np.full(nb_extra, 1)
    return X_extra, y_extra, W0, W1


def kmeans_lvq(n_prototypes, X, y, nb_extra=None, seed=1,
               append=True,
               kneighbours=30, number_epochs=10,
               data_boundary=True,
               sampling_strategy=1.0,
               learning_starting_rate=0.05,
               learning_rate_decrease=0.95,
               minimum_learning_rate=0.001,
               m=0.3,
               iter_max_fac=10,
               verbose=False):
    """
    Local LVQ learning by splitting the data. k-neighbours are found
    around (distance^2 metric). For each neighbourhood minority LVQ
    is used to find a local prototype for each class. n_prototypes
    prototype-pairs are created. To syntheize new data, random values are
    computed and the closest distance prototype-pair is chosen and the
    distance between the two classes is used to accept the the new values as
    new data.
    The strategy is unlike lvq_prototypes where the splitting is done by
    randomly. Both strategies are devised to capture local variations in
    the N-dimension.

    Parameters
    ----------
    n_prototypes : int
        the number of prototype per class.
        n_prototypes = 2 means 2 for each class in y

    X : array-like
        original input data set scaled to values beween 0 and 1

    y : array-like
        original input classes

    kneighbours : int, optional, default=30
        the number of neighbouring points to use for the local learning

    number_epochs : int, optional, default = 10
        the number of epochs used for the lvq training

    seed : int, optional, default = 1
        random generator seed value

    append : boolean, optional, default=True
        append the augmented data to the input

    nb_extra : int, optional, default=None
        the number of extra minority data point to be generated
        if None, the diffference in the input between the two
        classes is used together with the value of sampling_strategy

    sampling_strategy : float, optional, default=1.0
        the fraction between the two classes.
        sampling_strategy = 1. means that both classes will be balanced

    verbose : boolean, optional, default=False
        True to have screen logging

    data_boundary : boolean, optional, default=True
        if True, bound the augmented data within the range of the input

    learning_starting_rate : float, optional, default=0.05
        the starting value of the learning rate for the LVQ3

    learning_rate_decrease : float, optional, default=0.95
        the learning rate decrease factor (< 1)

    minimum_learning_rate : float, optional, default=0.001
        the minimum learning rate

    m : float, optional, default=0.3
        a damping factor for the learning between 0 < m < 1

    iter_max_fac : optinal, int, default=10
        the number of attempts to find a new valid data per required
        new data

    Returns
    -------
    X_extra : list
        list of original + extra datasets
        one entry per prototype

    y_extra : list
        list of original + extra corresponding classes

    W0: list
        the weights for the class 0

    W1 : list
        the weights for the class 1

    Example
    -------
    >>> import numpy as np
    >>> from collections import Counter
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_moons, make_circles
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from imblearn.datasets import make_imbalance
    >>> from imblearn.over_sampling import SMOTE
    >>> from scipy.spatial import ConvexHull, convex_hull_plot_2d
    >>> from LVQImbalance import kmeans_lvq
    >>> Xc, yc = make_circles(n_samples=600, noise=0.2, factor=0.5,
    ...                       random_state=0)
    >>> sampling_strategy = {1: 50}
    >>> X_circ, y_circ = make_imbalance(Xc, yc,
    ...                               sampling_strategy=sampling_strategy)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X_circ)
    >>> Xs = scaler.transform(X_circ)
    >>> n_prototypes = 10
    >>> count = Counter(y_imb)
    >>> nb_extra = count[0] - count[1] - n_prototypes * 2
    >>> X_lvq, y_lvq, W0, W1, W0init, W1init, cntr =\
    ...    kmeans_lvq(n_prototypes, Xs, y_circ,
    ...               number_epochs=10,
    ...               nb_extra=nb_extra)
    >>> X_extra = np.vstack((X_lvq, W1, cntr))
    >>> y_extra = np.append(y_lvq, np.full(2 * n_prototypes, 1))
    >>> hull0 = ConvexHull(W0)
    >>> wc = y_circ == 0
    >>> size = 10
    >>> plt.scatter(X_lvq[:, 0], X_lvq[:, 1], s=20, label='LVQ')
    >>> plt.scatter(Xs[wc, 0], Xs[wc, 1], s=size, label='Majority')
    >>> plt.scatter(Xs[~wc, 0], Xs[~wc, 1], s=size, label='Minority')
    >>> plt.scatter(W0[:, 0], W0[:, 1], s=50, marker='*', label='Prototype 0')
    >>> plt.scatter(W1[:, 0], W1[:, 1], s=50, marker='*', label='Prototype 1')
    >>> plt.scatter(cntr[:, 0], cntr[:, 1], s=50, marker='+',
    ...             label='kmeans cntr')
    >>> for simplex in hull0.simplices:
    >>>     plt.plot(W0[simplex, 0], W0[simplex, 1], 'k-')
    >>> plt.xlabel('X1')
    >>> plt.ylabel('X2')
    >>> plt.legend()
    >>> plt.tight_layout()
    >>> plt.show()
    >>> #
    >>> # create the Moon sample
    >>> Xmoon, ymoon = make_moons(n_samples=600, noise=0.2, random_state=0)
    >>> sampling_strategy = {1: 50}
    >>> X_imb, y_imb = make_imbalance(Xmoon, ymoon,
    ...                               sampling_strategy=sampling_strategy)
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X_imb)
    >>> Xs = scaler.transform(X_imb)
    >>> n_prototypes = 10
    >>> count = Counter(y_imb)
    >>> nb_extra = count[0] - count[1] - n_prototypes
    >>> X_lvq, y_lvq, W0, W1, W0init, W1init, cntr =\
    ...    kmeans_lvq(n_prototypes, Xs, y_imb,
    ...               number_epochs=10,
    ...               nb_extra=nb_extra)
    >>> X_extra = np.vstack((X_lvq, W1))
    >>> y_extra = np.append(y_lvq, np.full(n_prototypes, 1))
    >>> w = y_imb == 0
    >>> size = 10
    >>> plt.scatter(W0init[:, 0], W0init[:, 1],
    ...             s=50, marker='o', label='Init 0')
    >>> plt.scatter(W1init[:, 0], W1init[:, 1],
    ...             s=50, marker='o', label='Init 1')
    >>> plt.scatter(X_lvq[:, 0], X_lvq[:, 1], s=20, label='LVQ')
    >>> plt.scatter(Xs[w, 0], Xs[w, 1], s=size, label='Majority')
    >>> plt.scatter(Xs[~w, 0], Xs[~w, 1], s=size, label='Minority')
    >>> plt.scatter(W0[:, 0], W0[:, 1], s=50, marker='*', label='Prototype 0')
    >>> plt.scatter(W1[:, 0], W1[:, 1], s=50, marker='*', label='Prototype 1')
    >>> plt.scatter(cntr[:, 0], cntr[:, 1], s=50, marker='+',
    ...             label='kmeans cntr')
    >>> plt.xlabel('X1')
    >>> plt.ylabel('X2')
    >>> plt.legend()
    >>> plt.tight_layout()
    >>> plt.show()
    >>> # SMOTE + Tree-LVQ
    >>> sm = SMOTE(sampling_strategy=0.5, random_state=1235)
    >>> X_res, y_res = sm.fit_resample(Xs, y_imb)
    >>> n_prototypes = 10
    >>> count = Counter(y_res)
    >>> nb_extra = count[0] - count[1] - n_prototypes
    >>> X_lvq, y_lvq, W0, W1, W0init, W1init, cntr = \
    ...    kmeans_lvq(n_prototypes, X_res, y_res,
    ...               append=False,
    ...               nb_extra=nb_extra,
    ...               verbose=True)
    >>> X_extra = np.vstack((X_lvq, W1))
    >>> y_extra = np.append(y_lvq, np.full(n_prototypes, 1))
    >>> w = y_imb == 0
    >>> size = 10
    >>> plt.scatter(W0init[:, 0], W0init[:, 1],
    ...             s=50, marker='o', label='Init 0')
    >>> plt.scatter(W1init[:, 0], W1init[:, 1],
    ...             s=50, marker='o', label='Init 1')
    >>> plt.scatter(X_res[:, 0], X_res[:, 1], s=20, label='SMOTE')
    >>> plt.scatter(X_lvq[:, 0], X_lvq[:, 1], s=20, label='LVQ')
    >>> plt.scatter(Xs[w, 0], Xs[w, 1], s=size, label='Majority')
    >>> plt.scatter(Xs[~w, 0], Xs[~w, 1], s=size, label='Minority')
    >>> plt.scatter(W0[:, 0], W0[:, 1], s=50, marker='*', label='Prototype 0')
    >>> plt.scatter(W1[:, 0], W1[:, 1], s=50, marker='*', label='Prototype 1')
    >>> plt.scatter(cntr[:, 0], cntr[:, 1], s=50, marker='+',
    ...             label='kmeans cntr')
    >>> plt.xlabel('X1')
    >>> plt.ylabel('X2')
    >>> plt.legend()
    >>> plt.tight_layout()
    >>> plt.show()
    """
    assert X.shape[0] > kneighbours
    count = Counter(y)
    if count[0] > count[1]:
        majority = 0
        minority = 1
    else:
        majority = 1
        minority = 0
    wminority = np.where(y == minority)[0]
    wmajority = np.where(y == majority)[0]
    treeMinority = BallTree(X[wminority, :], leaf_size=2)
    treeMajority = BallTree(X[wmajority, :], leaf_size=2)
    kmeans = KMeans(n_clusters=n_prototypes,
                    random_state=seed).fit(X[y == minority, :])
    W0, W1, W0init, W1init = [], [], [], []
    ly = len(wminority)
    for i, cntr in enumerate(kmeans.cluster_centers_):
        if verbose:
            print('Prototype:', i + 1)
        # Find the closest minority instance to one of the Kmean
        # cluster center
        indMinority = treeMinority.query(cntr.reshape(1, -1),
                                         k=int(ly / n_prototypes) + 1,
                                         sort_results=True,
                                         return_distance=False)[0]
        iminority = wminority[indMinority[0]]
        indMajority = treeMajority.query(cntr.reshape(1, -1),
                                         k=kneighbours + 1,
                                         sort_results=True,
                                         return_distance=False)[0]
        imajority = wmajority[indMajority[0]]
        Winit = np.stack((X[imajority, :], X[iminority, :]))
        W0init.append(Winit[0])
        W1init.append(Winit[1])
        ly = len(y)
        mask = np.zeros(ly, dtype=bool)
        mask[wminority[indMinority[1:]]] = True
        mask[wmajority[indMajority[1:]]] = True
        X_train = X[mask, :]
        y_train = y[mask]
        # 0.1 < m < 0.5 is a stabilizing constant.
        counter_y = Counter(y_train)
        cy = counter_y[minority]
        if verbose:
            print("Number of minority sample",  cy)
        W = lvq3_train(X_train, y_train, Winit,
                       learning_starting_rate,
                       learning_rate_decrease,
                       number_epochs,
                       minimum_learning_rate,
                       m,
                       verbose=verbose)
        W0.append(W[0][0])
        W1.append(W[0][1])
    W0 = np.array(W0)
    W1 = np.array(W1)
    # synthetic data generation
    if nb_extra is None:
        nb_extra = int(np.ceil(sampling_strategy * count[majority] -
                               count[minority]))
    wpos = y == minority
    Xmin = X[wpos, :].min(0)
    Xmax = X[wpos, :].max(0)
    iter = 0
    npos = 0
    iter_max = iter_max_fac * nb_extra
    pos = []
    nrnd = X.shape[1]
    while npos < nb_extra and iter < iter_max:
        iter += 1
        rng = np.random.default_rng(seed + iter)
        rnd = rng.random(nrnd)
        if data_boundary:
            rnd = rnd * (Xmax - Xmin) + Xmin
        d0 = np.sum((rnd - W0)**2, 1)
        d1 = np.sum((rnd - W1)**2, 1)
        # find the closest prototype pairs and compare
        # the distances to them
        imin = np.argmin(d0 + d1)
        if np.mean((d1[imin] < d0[imin]) * 1) > 0.5:
            pos.append(rnd)
            npos += 1
    if iter >= iter_max:
        print("Not enough synthetic data generated.")
        print("Please increase iter_max_fac")
    pos = np.array(pos)
    if append:
        X_extra = np.vstack((X, pos))
        y_extra = np.append(y, np.full(nb_extra, 1))
    else:
        X_extra = pos
        y_extra = np.full(nb_extra, 1)
    return X_extra, y_extra, W0, W1, np.array(W0init), np.array(W1init), \
        kmeans.cluster_centers_
