from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaboostClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self):
        super(AdaboostClassifier, self).__init__()
        self.n_estimators = None
        self.learning_rate = None
        self.algorithm = None
        self.random_state = None
        self.max_depth = None
        self.estimator = None

    def fit(self, X, Y, sample_weight=None):

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)

        estimator = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )

        estimator.fit(X, Y, sample_weight=sample_weight)

        self.estimator = estimator
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
        return {'shortname': 'AB',
                'name': 'AdaBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, SIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # base_estimator = Constant(name="base_estimator", value="None")
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default=50, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default=0.1, log=True)
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["SAMME.R", "SAMME"], default="SAMME.R")
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=1, log=False)

        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(learning_rate)
        cs.add_hyperparameter(algorithm)
        cs.add_hyperparameter(max_depth)

        return cs

