# LearningVectorQuantizationClassifier
The Learning vector quantization (LVQ) is a prototype-based supervised   classification algorithm proposed by T. Kohonen.
earning Vector Quantization is a machine learning classifying algorithm (Wikipedia). Here is a scikit-learn compatible routine .

The scikit-learn-compatible interface allows you to use LVQ models just like any scikit-learn model in Python. Examples are provided in the notebook below. Learning vector quantization (LVQ) is a prototype-based supervised classification algorithm proposed by T. Kohonen. This type of neural network is used for classification. It is sometimes called a self-organizing neural net. It iteratively classifies inputs, until the combined difference between classes is maximized. This algorithm can be used as a simple way to cluster data, if the number of cases or categories is not particularly large.

An advantage of LVQ is that it creates prototypes that are easy to interpret for experts in the domain of application. A major disadvatange is that for data sets with a large number of categories, training the network can take a very long time.

Two training methods are available. By default the standard LVQ training is used. LVQ2 training scheme is  used withen the flag LVQ2 is set to True. A variant of the Neural Gas scheme can be used instead of the standard LVQ scheme.

A single-prototype LVQ method is provided to augment minority class data in imblanced datasets.

#### Reference

T. Kohonen. Self-Organizing Maps. Springer, Berlin, 1997.

Martinez, T., Berkovich, G. and Schulten, K.J.: Neural Gas Network for Vector Quantization and its Application to Time-Series Prediction. In: IEEE Transactions on Neural Networks, 4, 4, 558- 569 (1993)

Here is a comparison of the LVQ method as a classifier with other methods.

![LVQ_classifier_comparison](https://github.com/wfthi/LearningVectorQuantizationClassifier/assets/94956037/aec4c3da-9106-4c5a-860d-3175bd1601d6)

# LVQ imbalance dataset resampling

For binary classification, the occurance of the often interesting class is very small (minority class). The imbalance between this minority class and the other class (majority class) can hamper the efficiency of machine learning algorithms. A widely used method to resample the minority class is the Synthetic Minority Oversampling Technique (SMOTE) method. Here I made use of the prototype-learning of the LVQ method to generate new samples of the minority class (data augmentation). I show the resampling using the SMOTE default method from sklearn-imbalance and the LVQ method. After resampling, a standard classification algorithm like the Random Forrest algorithm can be applied.

![smote_resampling](https://github.com/wfthi/LearningVectorQuantizationClassifier/assets/94956037/18303f9c-dfeb-4626-bbae-97ca6e6743e9)

![LVQresampling](https://github.com/wfthi/LearningVectorQuantizationClassifier/assets/94956037/92ae9dfa-38c5-477f-a2eb-aff1e0e22bd7)

### Reference

N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.

