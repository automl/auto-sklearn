import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class ExtraTreesPreprocessorRegression(AutoSklearnPreprocessingAlgorithm):

    def __init__(self):
        self.n_estimators = None
        self.estimator_increment = 10
        self.criterion = None
        self.max_leaf_nodes = None
        self.max_depth = None
        self.min_samples_leaf = None
        self.min_samples_split = None
        self.max_features = None
        self.bootstrap = False
        self.oob_score = None
        self.n_jobs = 1
        self.random_state = None
        self.verbose = 0
        super(ExtraTreesPreprocessorRegression, self).__init__()

    def fit(self, X, Y):
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import SelectFromModel

        num_features = X.shape[1]
        max_features = int(
            float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        estimator = ExtraTreesRegressor(
            n_estimators=self.n_estimators, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
            max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state)

        estimator.fit(X, Y)
        self.preprocessor = SelectFromModel(estimator=estimator,
                                            threshold='mean',
                                            prefit=True)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ETR',
                'name': 'Extra Trees Regressor Preprocessing',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, SIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = cs.add_hyperparameter(Constant("n_estimators", 100))
        criterion = cs.add_hyperparameter(Constant("criterion", "mse"))
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
