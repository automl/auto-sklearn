from ParamSklearn.classification import ParamSklearnClassifier
from HPOlibConfigSpace.random_sampler import RandomSampler
import sklearn.datasets
import sklearn.metrics
import numpy as np

iris = sklearn.datasets.load_iris()
X = iris.data
Y = iris.target
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
configuration_space = ParamSklearnClassifier.get_hyperparameter_search_space()
sampler = RandomSampler(configuration_space, 1)
for i in range(10000):
    configuration = sampler.sample_configuration()
    auto = ParamSklearnClassifier(configuration)
    try:
        auto = auto.fit(X[indices[:100]], Y[indices[:100]])
    except Exception as e:
        print configuration
        print e
        continue
    predictions = auto.predict(X[indices[100:]])
    print sklearn.metrics.accuracy_score(predictions, Y[indices[100:]])