First Steps with AutoSklearn
****************************

This example demonstrates how to get the whole search space covered by
AutoSklearn, feed it to the random search algorithm implemented by the hyperopt
package and then train a classifier with a random configuration on the iris 
dataset.

    >>> from AutoSklearn.autosklearn import AutoSklearnClassifier
    >>> import sklearn.datasets
    >>> import sklearn.metrics
    >>> import numpy as np
    >>> import hyperopt
    >>> iris = sklearn.datasets.load_iris()
    >>> X = iris.data
    >>> Y = iris.target
    >>> indices = np.arange(X.shape[0])
    >>> np.random.shuffle(indices)
    >>> auto = AutoSklearnClassifier()
    >>> search_space = auto.get_hyperparameter_search_space()
    >>> configuration = hyperopt.pyll.stochastic.sample(search_space)
    >>> auto = AutoSklearnClassifier(classifier=configuration['classifier'], preprocessor=configuration['preprocessing'])
    >>> auto = auto.fit(X[indices[:100]], Y[indices[:100]])
    >>> predictions = auto.predict(X[indices[100:]])
    >>> sklearn.metrics.accuracy_score(predictions, Y[indices[100:]])
    
