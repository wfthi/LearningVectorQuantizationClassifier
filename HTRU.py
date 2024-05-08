"""
    HTRU2 is a data set which describes a sample of pulsar candidates
    collected during the High Time Resolution Universe Survey (South

    Dataset:
    Author: Rob Lyon, School of Computer Science & Jodrell Bank Centre
    for Astrophysics,
    University of Manchester, Kilburn Building, Oxford Road,
    Manchester M13 9PL.

    1. Mean of the integrated profile.
    2. Standard deviation of the integrated profile.
    3. Excess kurtosis of the integrated profile.
    4. Skewness of the integrated profile.
    5. Mean of the DM-SNR curve.
    6. Standard deviation of the DM-SNR curve.
    7. Excess kurtosis of the DM-SNR curve.
    8. Skewness of the DM-SNR curve.
"""
# Author: Wing-Fai Thi
# License: BSD
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from astroML.classification import GMMBayes
from astroML.utils import split_samples, completeness_contamination
# XGBoost
from xgboost.sklearn import XGBClassifier  # use the XGBoost routine
from sklearn.metrics import classification_report
from LVQClassifier import LVQClassifier

# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
import seaborn as sns; sns.set()
setup_text_plots(fontsize=12, usetex=True)

# get data and split into training & testing sets
# fetch dataset from UCI
print("Fetching the data")
htru2 = fetch_ucirepo(id=372)
# data (as pandas dataframes)
X = np.array(htru2.data.features)
y = np.array(htru2.data.targets).astype(int)
y = y.astype(int).flatten()

# Select the columns to be used in the analysis
df_data = pd.DataFrame(X)
# Basic correlogram using seaborn
df_data.rename(columns=htru2.variables.name, inplace=True)
sns.pairplot(df_data, height=1.0)
plt.show()

title = 'Pulsar detection'
print("Number of instances (objects) & number of features:", X.shape)
(X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.75, 0.25],
                                                     random_state=0)
n_features = X_train.shape[1]

# Standard scaling of the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# PCA transform
pca = PCA(n_components=n_features)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# oversampling the RRLyrae in the training set
X_train_rr = X_train[y_train == 1]
k = 10
# multiply by 1k the number of positive in the training set!
for i in range(k):
    X_train = np.vstack((X_train, X_train_rr))
    y_train = np.concatenate((y_train, np.full(X_train_rr.shape[0], 1.)))
# completeness astroML = recall tp/(tp+fn) sklearn.metrics.recall_score

# Fit all the models to the training data

max_features = int(np.sqrt(n_features))
classifiers = {'GNB': GaussianNB(),
               'LDA': LDA(),
               'QDA': QDA(),
               'LR': LogisticRegression(class_weight='balanced',
                                        penalty='l1', solver='saga', tol=0.1),
               'KNN': KNeighborsClassifier(n_neighbors=15 * (k + 1)),
               'DT': DecisionTreeClassifier(random_state=10,
                                            max_depth=max_features,
                                            criterion='gini',
                                            class_weight='balanced'),
               'ExT': ExtraTreesClassifier(random_state=10,
                                           max_depth=max_features,
                                           criterion='entropy',
                                           class_weight='balanced'),
               'RF': RandomForestClassifier(n_estimators=100,
                                            random_state=10,
                                            max_depth=max_features,
                                            criterion='gini',
                                            class_weight='balanced'),
               'Ada': AdaBoostClassifier(n_estimators=100,
                                         learning_rate=0.1,
                                         random_state=2017,
                                         algorithm='SAMME'),
               'GMMBayes': GMMBayes(n_components=3),
               'XGB': XGBClassifier(n_estimators=100, max_depth=3,
                                    objective='binary:logistic', seed=2001,
                                    learning_rate=0.1),
               'NN': MLPClassifier(solver='adam',
                                   alpha=1e-5,
                                   hidden_layer_sizes=(100, 100),
                                   random_state=1,
                                   early_stopping=True,
                                   shuffle=False,
                                   batch_size='auto',
                                   validation_fraction=0.1,
                                   verbose=False),
               'SVM': SVC(kernel='rbf',
                          class_weight='balanced',
                          C=1.0,
                          gamma=2,
                          probability=True),
               'LVQ': LVQClassifier(n_components=500,
                                    alpha=0.1,
                                    epochs=15,
                                    p=1.,
                                    LVQ2=True,
                                    random_state=2018,
                                    initial_state='Normal')}

# SVM noisy data decrease C (default 1), use class_weith='balanced'
# 'GP':    GaussianProcessClassifier(1.0 * RBF(1.0))}

filename = 'pulsar_NTRU2_ROC.sav'
clfs = []
# read previously run models using joblib from sklearn
if os.path.exists(filename):
    print("Read previously ran models")
    clfs = joblib.load(filename)

names = []
probs = []
preds = []
F1 = []
target_names = ['not pular', 'pulsar']

for i, (name, classifier) in enumerate(classifiers.items()):
    print(name, classifier)
    if not (os.path.exists(filename)):
        model = classifier
        if name == 'XGB':
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        else:
            model.fit(X_train, y_train)
    else:
        model = clfs[i]
    if (name == 'SVM'):
        y_prob = np.exp(model.predict_log_proba(X_test))
    else:
        y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    probs.append(y_prob[:, 1])
    preds.append(y_pred)
    F1.append(f1_score(y_pred, y_test))
    names.append(name)
    clfs.append(model)
    rp = classification_report(y_test, y_pred, target_names=target_names)
    print(rp)

# Save the model to disk
if not (os.path.exists(filename)):
    print("Saving models")
    joblib.dump(clfs, filename)

for (name, classifier), F1 in zip(classifiers.items(), F1):
    print(name, " - F1 score:", F1)

# Plot ROC curves and completeness/efficiency
color = ['blue', 'silver', 'darkgoldenrod', 'darkgreen', 'darkmagenta',
         'pink', 'darkorange', 'gold', 'darkorchid', 'aqua', 'cadetblue',
         'darkolivegreen', 'black', 'green', 'red']

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25)

# ax2 will show roc curves
ax1 = plt.subplot(121)

# ax1 will show completeness/efficiency
ax2 = plt.subplot(122)

thresholds = np.linspace(0, 1, 1001)[:-1]

# iterate through and show results
for j, (name, y_prob) in enumerate(zip(names, probs)):
    fpr, tpr, thresh = roc_curve(y_test, y_prob)

    # add (0, 0) as first point
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])

    ax1.plot(fpr, tpr, label=name, c=color[j])

    comp = np.zeros_like(thresholds)
    cont = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t)
        comp[i], cont[i] = completeness_contamination(y_pred, y_test)
    ax2.plot(1 - cont, comp, label=name, c=color[j])

ax1.set_title("Receiver Operating Characteristic (ROC)")
ax1.set_xlim(0, 0.04)
ax1.set_ylim(0, 1.02)
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
ax1.legend(loc=4)

ax2.set_xlabel('efficiency (1-contamination)')
ax2.set_ylabel('completeness')
ax2.set_title(title)
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0.2, 1.02)

plt.savefig("HTRU2_ROC_curve.pdf")
plt.show()

# Second figure: calibration plots
plt.figure(figsize=(10, 10))
ax3 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax4 = plt.subplot2grid((3, 1), (2, 0))
ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for j, (name, y_prob) in enumerate(zip(names, probs)):
    print(name, y_prob.min(), y_prob.max())
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test,
                                                                    y_prob,
                                                                    n_bins=10)
    ax3.plot(mean_predicted_value, fraction_of_positives,
             "s-", label="%s" % (name, ))
    ax4.hist(y_prob, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)
ax3.set_ylabel("Fraction of positives")
ax3.set_xlabel("mean predicted value")
ax3.set_ylim([-0.05, 1.05])
ax3.legend(loc="upper left")
ax3.set_title('Calibration plots (reliability curve)')
ax4.set_xlabel("Mean predicted value")
ax4.set_ylabel("Count")
ax4.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig("HTRU2_calibration_curve.pdf")
plt.show()
