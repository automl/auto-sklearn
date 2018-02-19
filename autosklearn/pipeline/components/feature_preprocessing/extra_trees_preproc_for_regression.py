import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none


class ExtraTreesPreprocessorRegression(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features,
                 bootstrap=False, max_leaf_nodes=None, max_depth="None",
                 min_weight_fraction_leaf=0.0,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0):

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        if criterion not in ("mse", "friedman_mse", "mae"):
            raise ValueError("'criterion' is not in ('mse', 'friedman_mse', "
                             "'mae'): %s" % criterion)
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf

        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.preprocessor = None

    def fit(self, X, Y):
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import SelectFromModel

        self.n_estimators = int(self.n_estimators)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.max_features = float(self.max_features)
        self.bootstrap = check_for_bool(self.bootstrap)
        self.n_jobs = int(self.n_jobs)
        self.verbose = int(self.verbose)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)

        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

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
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
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
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter("criterion",
                                              ["mse", 'friedman_mse', 'mae'])
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0)

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = Constant('min_weight_fraction_leaf', 0.)

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False")

        cs.add_hyperparameters([n_estimators, criterion, max_features, max_depth,
                                max_leaf_nodes, min_samples_split,
                                min_samples_leaf, min_weight_fraction_leaf,
                                bootstrap])

        return cs
