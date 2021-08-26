import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.utils.validation import check_random_state

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
)
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS


class GaussianNB(AutoSklearnClassificationAlgorithm):

    def __init__(self, random_state=None, verbose=0):
        self.verbose = int(verbose)
        self.estimator = None
        self.random_state = check_random_state(random_state)
        self._random_seed = self.random_state.randint(np.iinfo(np.uint32).max, dtype='u8')

    def fit(self, X, y):
        import sklearn.naive_bayes

        self.estimator = sklearn.naive_bayes.GaussianNB()
        self.classes_ = np.unique(y.astype(int))

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
        self.estimator.fit(X, y)

        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GaussianNB',
                'name': 'Gaussian Naive Bayes classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
