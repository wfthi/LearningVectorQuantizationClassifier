import numpy as np


class LVQClassifier():
    """
    Learning vector quantization (LVQ) is a prototype-based supervised
    classification algorithm proposed by T. Kohonen.

    This type of neural network is used for classification. It is sometimes
    called a self-organizing neural net. It iteratively classifies inputs,
    until the combined difference between classes is maximized.
    This algorithm can be used as a simple way to cluster data,
    if the number of cases or categories is
    not particularly large.

    A major disadvatange is that for data sets with a large number of
    categories, training the network can take a very long time.

    from Wikipedia:

    "LVQ can be understood as a special case of an artificial neural network,
    more precisely, it applies a winner-take-all Hebbian learning-based
    approach. It is a precursor to self-organizing maps (SOM) and related to
    neural gas and the k-nearest neighbor algorithm (k-NN).
    LVQ was invented by Teuvo Kohonen.[1]"

    "An LVQ system is represented by prototypes W=(w(i),...,w(n))} which
    are defined in the feature space of observed data. In
    winner-take-all training algorithms one determines, for each data
    point, the prototype which is closest to the input according to a given
    distance measure.
    The position of this so-called winner prototype is then adapted, i.e.
    the winner is moved closer if it correctly classifies the data point or
    moved away if it classifies the data point incorrectly.

    An advantage of LVQ is that it creates prototypes that are easy to
    interpret for experts in the respective application domain. LVQ
    systems can be applied to multi-class classification problems in a
    natural way. It is used in a variety of practical applications."

    Because of the nature of the the winner takes it all nature of the
    algorithm, to avoid never-winning prototypes a bias to the distances is
    applied.

    Two training methods are available. By default the standard LVQ training
    is used. LVQ2 training scheme is used withen the flag LVQ2 is set to True.

    Ref.

    T. Kohonen. Self-Organizing Maps. Springer, Berlin, 1997.

    Martinez, T., Berkovich, G. and Schulten, K.J.: Neural Gas Network for
        Vector Quantization and its Application to Time-Series Prediction.
        In: IEEE Transactions on Neural Networks, 4, 4, 558- 569 (1993)

    http://www.cis.hut.fi/panus/papers/dtwsom.pdf

    Author: Wing-Fai Thi, wingfai.thi@googlemail.com

    License: GNU verion 3.0

    History:
        Version 1.0 (28/2/2018)
        Version 1.1 (5/3/2018) add attributes and methods to improve
                               compatibility with scikit-learn
        Version 1.3 (12/3/2018) add Monte-Carlo probabily errors and
                                assertion tests
        Version 1.4 (21/3/2018) add the neural-gas option and correct
                                bug in the heade

    Package requirements:
    ---------------------
    numpy

    Parameters
    ----------
    n_components: integer (default=1)
        number of prototypes per class

    alpha : float, optional (default=0.5)
        learning rate constant

    decrease_rate : float, optional (default=0.9)
        learning rate decrease rate

    p : float, optional (default=2 Eucledean)
        The power of the Minkowski metric to be used to calculate
        distance between points.

    epochs: integer, optional (default=3)
        number of learning epoch

    random_state : integer, optioncal (default=1)
        seed of the random number generator

    learning_rate = alpha*(1.-decrease_rate*epoch/epochs)

    bias_decrease_rate : float, optional (default=0.9)
        to avoid not winning prototypes (dead neurons), the distances
        are modified by a bias not winning prototypes will have their
        distances decreased by bias_decrease_rate winning prototypes
        will have their distances increased by bias_decrease_rate to
        penalize their success

    LVQ2 : boolean (default=False)
        Use the LVQ2 training model:
        - positive match : the winner weights get closer to the training
            features
        - negative match : the winner weights go away from the training
            features the closest match weights get closer to the training
            features

    initial_state : string (default='Normal')
        distribution of the initial prototypes (Normal or Uniform)

    sigma : float (default='1')
        standard deviation around the average for the initial distribution
        if a normal distribution is requested. It has no effect if
        initial_state='Uniform'

    algorithm: string (default='LVQ')
        use the LVQ or the Neural Gas algorithm (algorithm='NeuralGas')

    lamb: float (default=0.1)
        parameter to control the learning decrease rate
        exp(-k/(lamb*number_of_prototypes). Used only when
        algorithm='NeuralGas'

    verbose : integer (default=0)
        level of screen output

    Attributes
    ----------
    n_classes_ : integer
        number of classes

    classes_: array of shape(n_classes_)
        labels of the classes

    weights : array of shape (n_components*2,n_features)
        weights of each prototypes

    label_weights : array of shape (n_components*2)
        label of each prototype

    wins : array of shape (n_components*2)
        number of wins of each prototype during the training

    Methods
    -------
    fit(X,y)
        Fit the model using X as training data and y as target values

        Parameters X array of shape(n_samples,n_features)

        Returns None

    predict(X)
        Returns the predicted label for X. the fit method has to been run
        before

        Parameters X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples

    predict_proba(X)
        Returns the predicted class probability for each instance in the
        input X

        Parameters X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples,number of classes)

    score((self, X, y, sample_weight=None)
        Returns the mean accuracy on the given test data and labels.

    get_params((self)
        Get parameters for this estimator

    set_params(self, **params)
        Set the parameters of this estimator.

    Example
    -------
    >>> import numpy as np
    >>> from LVQClassifier import LVQClassifier
    >>> X=np.array([[1.,3.],[3.,4.],[6.,1.],[8.,3.],[9.,1.],[1.,6.]])
    >>> y=np.array([0,0,1,1,1,0])
    >>> lvq = LVQClassifier(n_components=3,random_state=2018)
    >>> lvq.fit(X,y)
    >>> X_test=np.array([[3.,2.],[7.,4.],[5.,1.]])
    >>> y_test=[0,1,1]
    >>> pred = lvq.predict(X_test)
    >>> proba = lvq.predict_proba(X_test)
    >>> print(y_test)
    >>> print(pred)
    >>> print(proba)
    >>> X_LVQ = lvq.weights
    >>> y_LVQ = lvq.label_weights
    >>> all(y_test == pred)
    True
    """

    def __init__(self, n_components=1, alpha=0.5,
                 decrease_rate=0.9, epochs=3,
                 random_state=1,
                 p=2, bias_decrease_rate=0.9, LVQ2=False,
                 initial_state='Normal', sigma=1.,
                 algorithm='LVQ', lamb=0.1, verbose=0):
        """
        Learning Vector Classification
        Initialization of the default parameters
        """
        self.n_components = n_components
        self.alpha = alpha
        self.decrease_rate = decrease_rate
        # < 1 to avoid learning rate reaching 0
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        self.p = p  # Minkowski p=2 = Euclidean distance
        self.bias_decrease_rate = bias_decrease_rate
        self.LVQ2 = LVQ2
        self.initial_state = initial_state
        # can be otherwise uniform
        self.sigma = sigma
        self._estimator_type = "classifier"
        self.algorithm = algorithm
        self.lamb = lamb
        np.random.seed(random_state)
        assert n_components > 0
        assert alpha > 0
        assert decrease_rate > 0
        assert epochs > 0
        if p < 1:
            raise ValueError("p must be at least 1")
        assert bias_decrease_rate > 0
        if (algorithm not in ['LVQ', 'NeuralGas']):
            raise ValueError('algorithm can only be'
                             'LVQ (default) or NeuralGas')
        if (initial_state not in ['Normal', 'Uniform']):
            raise ValueError('initial_state not valid'
                             'Choose Normal or Uniform')

    def get_params(self):
        self.params = {'n_components': self.n_components,
                       'alpha': self.alpha,
                       'decrease_rate': self.decrease_rate,
                       'epochs': self.epochs,
                       'random_state': self.random_state,
                       'verbose': self.verbose,
                       'p': self.p,
                       'bias_decrease_rate': self.bias_decrease_rate,
                       'LVQ2': self.LVQ2,
                       'initial_state': self.initial_state,
                       'sigma': self.sigma,
                       'algorithm': self.algorithm,
                       'lamb': self.lamb}
        return self.params

    def set_params(self, **parameters):
        if not parameters:
            # direct return in no parameter
            return self
        valid_params = self.get_params()
        for parameter, value in parameters.items():
            if parameter not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (parameter, self.__class__.__name__))
            setattr(self, parameter, value)
        return self

    def train(self, X, y, w, wcl, X_err=None):
        nb_train = y.shape[0]
        n_neurons, n_features = w.shape
        bias = np.ones(n_neurons)
        wins = np.zeros(n_neurons)
        if (self.epochs == 0):
            return w, wins.astype(int)
        if (self.verbose > 0):
            print("LVQ2 training")
        for epoch in range(self.epochs):
            if (self.verbose > 0):
                print("epoch:", epoch+1, "/", self.epochs)
            learning_rate = self.alpha*(1.-self.decrease_rate * epoch /
                                        self.epochs)
            if (self.algorithm == 'NeuralGas'):
                lrate = learning_rate * np.exp(-np.arange(0, n_neurons, 1.) /
                                               (n_neurons * self.lamb))
                if (lrate.min() < 1e-4):
                    kmax = np.array(np.where(lrate < 1e-4)).min()
                else:
                    kmax = n_neurons
            for i in range(nb_train):
                distances = self.dist(X[i, :], w) * bias
                if (X_err is not None):
                    distances = distances / (1. +
                                             np.sqrt(np.sum(X_err[i, :]**2)) /
                                             n_features)
                ind = np.argsort(distances)
                min_cl = self.min_cl
                max_cl = self.max_cl
                if (self.algorithm == 'NeuralGas'):
                    for k in range(kmax):
                        j = ind[k]
                        wclk = y[i].astype(int)  # -1 away +1 closer
                        new_w = w[j, :] + (2. * (wcl[j] == wclk)-1.) *\
                            lrate[k] * (X[i, :]-w[j, :])
                        if np.all((new_w > min_cl[wclk, :]) &
                                  (new_w < max_cl[wclk, :])):
                            w[j, :] = new_w
                    j = ind[0]
                else:
                    j = ind[0]  # index of the winning prototype/neuron
                    wclj = wcl[j].astype(int)
                    if (wclj == y[i]):
                        new_w = w[j, :] + learning_rate*(X[i, :]-w[j, :])
                        if np.all((new_w > min_cl[wclj, :]) &
                                  (new_w < max_cl[wclj, :])):
                            w[j, :] = new_w
                    else:
                        new_w = w[j, :] - learning_rate*(X[i, :]-w[j, :])
                        # the winner is going away
                        if np.all((new_w > min_cl[wclj, :])
                                  & (new_w < max_cl[wclj, :])):
                            w[j, :] = new_w
                        if (self.LVQ2):
                            # find the closest positive loser
                            k = ind[np.argmax(wcl[ind] == y[i])]
                            wclk = y[i].astype(int)
                            new_w = w[k, :] + learning_rate*(X[i, :]-w[k, :])
                            if np.all((new_w > min_cl[wclk, :]) &
                                      (new_w < max_cl[wclk, :])):
                                w[k, :] = new_w
                wins[j] = wins[j] + 1
                bias = bias * self.bias_decrease_rate
                bias[j] = bias[j]/self.bias_decrease_rate
                if (bias.min() < 1e-2):
                    bias = bias / bias.max()
        # return the weights and the number of wins (integer)
        return w, wins.astype(int)

    def fit(self, X, y, X_err=None):
        """
        fit to the training set X with label y
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        n_components = self.n_components

        if n_samples != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        if (n_components > n_samples):
            raise ValueError("too many components requested")

        self.classes_ = np.unique(y)  # class unique labels
        ycl = self.classes_  # class unique labels
        self.n_classes_ = ycl.size
        nbc = self.n_classes_  # number of classes

        w = []  # weights of each "neuron"
        wcl = []  # class of each neuron
        prototypes = []
        label = []
        nbcl = (ycl.max()+1).astype(int)
        min_cl = np.zeros((nbcl, n_features))
        max_cl = np.zeros((nbcl, n_features))
        avg_cl = np.zeros((nbcl, n_features))
        std_cl = np.zeros((nbcl, n_features))
        for j in range(nbc):
            max_cl[j] = X[y == ycl[j]].max(axis=0)
            min_cl[j] = X[y == ycl[j]].min(axis=0)
            avg_cl[j] = np.median(X[y == ycl[j]], axis=0)
            std_cl[j] = X[y == ycl[j]].std(axis=0)
        d_cl = max_cl-min_cl
        self.min_cl = min_cl
        self.max_cl = max_cl
        self.avg_cl = avg_cl
        self.std_cl = std_cl
        for i in range(n_components):
            for j in range(nbc):
                if (self.initial_state == 'Normal'):
                    # random starting points
                    w.append(np.random.normal(avg_cl[j, :],
                                              self.sigma*std_cl[j, :]))
                else:
                    # random starting points
                    w.append(np.random.random(n_features) *
                             d_cl[j, :] + min_cl[j, :])
                wcl.append(ycl[j])
        w = np.array(w)  # w = array(n_components * nbc, n_features)
        wcl = np.array(wcl)
        prototypes, self.wins = self.train(X, y, w, wcl, X_err=X_err)
        label.append(wcl)
        self.weights = np.array(prototypes).reshape(n_components *
                                                    nbc, n_features)
        self.label_weights = np.array(label).reshape(n_components*nbc)
        return self

    def dist(self, u, v):
        """
        The Minkowski distance between vectors `u` and `v`.
        """
        w = u-v
        if (self.p == 1):
            d = abs(w).sum(axis=1)
        if (self.p == 2):
            d = np.sqrt(np.power(w, 2).sum(axis=1))
        else:
            d = np.power(np.power(abs(w), self.p).sum(axis=1), 1. / self.p)
        return d

    def predict(self, X):
        """
        predict(self, X)

        Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            Class labels for each data sample.
        """
        n_samples, n_features = X.shape
        pred = []
        for i in range(n_samples):
            pred.append(self.label_weights[np.argmin(self.dist(X[i, :],
                                                               self.weights))])
        return np.array(pred)

    def predict_proba(self, X):
        """
        predict_proba(self, X)

        Return probability estimates for the test data X.

        Parameters
         ----------
        X : array-like, shape (n_query, n_features)

        Returns
        -------
        p : array of shape = [n_samples, n_classes_]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        n_samples, n_features = X.shape
        probabilities = np.empty((n_samples, self.n_classes_))
        for i in range(n_samples):
            distances = self.dist(X[i, :], self.weights)
            ind = np.argsort(distances)
            distances = distances[ind]
            label_list = self.label_weights[ind]
            proba_class = np.empty(self.n_classes_)
            for label in range(self.n_classes_):
                proba_class[label] = 1. /\
                    distances[np.argmax(label == label_list)]
            probabilities[i, :] = proba_class / proba_class.sum()
        return np.array(probabilities)

    def predict_proba_MC_error(self, X, X_err, Number_of_MC_iterations=1000):
        """
        Returns the Monte-Carlo errors on the probabilities added 12/3/2018
        """
        n_samples, n_features = X.shape
        proba_MC = []
        for i in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X, X_err)
            for j in range(n_samples):
                distances = self.dist(X_MC[j, :], self.weights)
                ind = np.argsort(distances)
                distances = distances[ind]
                label_list = self.label_weights[ind]
                proba_class = np.empty(self.n_classes_)
                for label in range(self.n_classes_):
                    proba_class[label] = 1. / \
                        distances[np.argmax(label == label_list)]
                proba_MC.append(proba_class/proba_class.sum())
        proba_MC = np.array(proba_MC).reshape(Number_of_MC_iterations,
                                              self.n_classes_, n_samples)
        return np.std(proba_MC, axis=0).T

    def score(self, X, y_true, sample_weight=None):
        """
        Returns the accuracy: tp+tn/(tp+tn+fp+fn)
        """
        n_samples, _ = X.shape
        n_true = y_true.shape[0]
        if (n_samples != n_true):
            raise ValueError("The sample and the label number of\
                             instances are different.")
        y_pred = self.predict(X)
        score = y_true == y_pred
        return self._weighted_sum(score, sample_weight, normalize=True)

    def _weighted_sum(self, sample_score, sample_weight,
                      normalize=False):  # similar to cikit-learn
        if normalize:
            return np.average(sample_score, weights=sample_weight)
        elif sample_weight is not None:
            return np.dot(sample_score, sample_weight)
        else:
            return sample_score.sum()

# -------------------------------------------------------------
