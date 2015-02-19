First Steps with ParamSklearn
*****************************

This example demonstrates how to get the whole configuration space covered by
ParamSklearn, feed it to the random search algorithm implemented by the
HPOlibConfigSpace package and then train a classifier with a random
configuration on the iris dataset.

    >>> from ParamSklearn.classification import ParamSklearnClassifier
    >>> from HPOlibConfigSpace.random_sampler import RandomSampler
    >>> import sklearn.datasets
    >>> import sklearn.metrics
    >>> import numpy as np
    >>> iris = sklearn.datasets.load_iris()
    >>> X = iris.data
    >>> Y = iris.target
    >>> indices = np.arange(X.shape[0])
    >>> np.random.seed(1)
    >>> np.random.shuffle(indices)
    >>> configuration_space = ParamSklearnClassifier.get_hyperparameter_search_space()
    >>> sampler = RandomSampler(configuration_space, 5)
    >>> configuration = sampler.sample_configuration()
    >>> cls = ParamSklearnClassifier(configuration, random_state=1)
    >>> cls = cls.fit(X[indices[:100]], Y[indices[:100]])
    >>> predictions = cls.predict(X[indices[100:]])
    >>> sklearn.metrics.accuracy_score(predictions, Y[indices[100:]])
    0.81999999999999995
