import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.pipeline.implementations.util import convert_multioutput_multiclass_to_multilabel


class ExtraTreesClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self):
        super(ExtraTreesClassifier, self).__init__()
        self.n_estimators = None
        self.estimator_increment = None
        self.criterion = None
        self.max_leaf_nodes = None
        self.max_depth = None
        self.min_samples_leaf = None
        self.min_samples_split = None
        self.max_features = None
        self.bootstrap = None
        self.oob_score = None
        self.n_jobs = None
        self.random_state = None
        self.verbose = None
        self.class_weight = None

    def fit(self, X, y, sample_weight=None, refit=False):
        if self.estimator is None or refit:
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight,
                               refit=refit)

        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight)
        return self

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import ExtraTreesClassifier as ETC

        if refit:
            self.estimator = None

        if self.estimator is None:
            num_features = X.shape[1]
            max_features = int(
                float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            self.estimator = ETC(
                n_estimators=0, criterion=self.criterion,
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
                max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
                oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
                random_state=self.random_state,
                class_weight=self.class_weight,
                warm_start=True
            )

        tmp = self.estimator  # TODO copy ?
        tmp.n_estimators += n_iter
        tmp.fit(X, y, sample_weight=sample_weight)
        self.estimator = tmp
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = cs.add_hyperparameter(Constant("n_estimators", 100))
        criterion = cs.add_hyperparameter(CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini"))
        max_features = cs.add_hyperparameter(UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1))

        max_depth = cs.add_hyperparameter(
            UnParametrizedHyperparameter(name="max_depth", value="None"))

        min_samples_split = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2))
        min_samples_leaf = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1))
        min_weight_fraction_leaf = cs.add_hyperparameter(Constant(
            'min_weight_fraction_leaf', 0.))

        bootstrap = cs.add_hyperparameter(CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default="False"))

        return cs
