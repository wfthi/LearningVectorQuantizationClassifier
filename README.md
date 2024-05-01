# LearningVectorQuantizationClassifier
The Learning vector quantization (LVQ) is a prototype-based supervised   classification algorithm proposed by T. Kohonen.
earning Vector Quantization is a machine learning classifying algorithm (Wikipedia). Here is a scikit-learn compatible routine .

The scikit-learn-compatible interface allows you to use LVQ models just like any scikit-learn model in Python. Examples are provided in the notebook below. Learning vector quantization (LVQ) is a prototype-based supervised classification algorithm proposed by T. Kohonen. This type of neural network is used for classification. It is sometimes called a self-organizing neural net. It iteratively classifies inputs, until the combined difference between classes is maximized. This algorithm can be used as a simple way to cluster data, if the number of cases or categories is not particularly large.

An advantage of LVQ is that it creates prototypes that are easy to interpret for experts in the domain of application. A major disadvatange is that for data sets with a large number of categories, training the network can take a very long time.

Two training methods are available. By default the standard LVQ training is used. LVQ2 training scheme is  used withen the flag LVQ2 is set to True. A variant of the Neural Gas scheme can be used instead of the standard LVQ scheme.

Reference

T. Kohonen. Self-Organizing Maps. Springer, Berlin, 1997.

Martinez, T., Berkovich, G. and Schulten, K.J.: Neural Gas Network for Vector Quantization and its Application to Time-Series Prediction. In: IEEE Transactions on Neural Networks, 4, 4, 558- 569 (1993)
