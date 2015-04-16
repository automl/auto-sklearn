import sklearn.lda

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from ParamSklearn.components.classification_base import \
    ParamSklearnClassificationAlgorithm
from ParamSklearn.util import SPARSE, DENSE, PREDICTIONS


class LDA(ParamSklearnClassificationAlgorithm):
    def __init__(self, n_components, tol, random_state=None):
        self.n_components = int(n_components)
        self.tol = float(tol)
        self.estimator = None

    def fit(self, X, Y):

        self.estimator = sklearn.lda.LDA(n_components=self.n_components)
        self.estimator.fit(X, Y, tol=self.tol)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.predict_proba(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, ),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformIntegerHyperparameter('n_components', 1, 250,
                                                    default=10)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default=1e-4,
                                         log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_components)
        cs.add_hyperparameter(tol)
        return cs
