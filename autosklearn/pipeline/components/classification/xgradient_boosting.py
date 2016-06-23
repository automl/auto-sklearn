import numpy

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm

from autosklearn.pipeline.constants import *


class XGradientBoostingClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, learning_rate, n_estimators, subsample,
                 max_depth, colsample_bylevel, colsample_bytree, gamma,
                 min_child_weight, max_delta_step, reg_alpha, reg_lambda,
                 base_score, scale_pos_weight, nthread=1, init=None,
                 random_state=None, verbose=0):
        ## Do not exist
        # self.loss = loss
        # self.min_samples_split = min_samples_split
        # self.min_samples_leaf = min_samples_leaf
        # self.min_weight_fraction_leaf = min_weight_fraction_leaf
        # self.max_leaf_nodes = max_leaf_nodes

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth

        ## called differently
        # max_features: Subsample ratio of columns for each split, in each level.
        self.colsample_bylevel = colsample_bylevel

        # min_weight_fraction_leaf: Minimum sum of instance weight(hessian)
        # needed in a child.
        self.min_child_weight = min_child_weight

        # Whether to print messages while running boosting.
        if verbose:
            self.silent = False
        else:
            self.silent = True

        # Random number seed.
        if random_state is None:
            self.seed = numpy.random.randint(1, 10000, size=1)[0]
        else:
            self.seed = random_state.randint(1, 10000, size=1)[0]

        ## new paramaters
        # Subsample ratio of columns when constructing each tree.
        self.colsample_bytree = colsample_bytree

        # Minimum loss reduction required to make a further partition on a leaf
        # node of the tree.
        self.gamma = gamma

        # Maximum delta step we allow each tree's weight estimation to be.
        self.max_delta_step = max_delta_step

        # L2 regularization term on weights
        self.reg_alpha = reg_alpha

        # L1 regularization term on weights
        self.reg_lambda = reg_lambda

        # Balancing of positive and negative weights.
        self.scale_pos_weight = scale_pos_weight

        # The initial prediction score of all instances, global bias.
        self.base_score = base_score

        # Number of parallel threads used to run xgboost.
        self.nthread = nthread

        ## Were there before, didn't touch
        self.init = init
        self.estimator = None

    def fit(self, X, y):
        import xgboost as xgb

        self.learning_rate = float(self.learning_rate)
        self.n_estimators = int(self.n_estimators)
        self.subsample = float(self.subsample)
        self.max_depth = int(self.max_depth)

        # (TODO) Gb used at most half of the features, here we use all
        self.colsample_bylevel = float(self.colsample_bylevel)

        self.colsample_bytree = float(self.colsample_bytree)
        self.gamma = float(self.gamma)
        self.min_child_weight = int(self.min_child_weight)
        self.max_delta_step = int(self.max_delta_step)
        self.reg_alpha = float(self.reg_alpha)
        self.reg_lambda = float(self.reg_lambda)
        self.nthread = int(self.nthread)
        self.base_score = float(self.base_score)
        self.scale_pos_weight = float(self.scale_pos_weight)

        # We don't support multilabel, so we only need 1 objective function
        if len(numpy.unique(y) == 2):
            # We probably have binary classification
            self.objective = 'binary:logistic'
        else:
            self.objective = 'multi:softprob'

        self.estimator = xgb.XGBClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                silent=self.silent,
                objective=self.objective,
                nthread=self.nthread,
                gamma=self.gamma,
                scale_pos_weight=self.scale_pos_weight,
                min_child_weight=self.min_child_weight,
                max_delta_step=self.max_delta_step,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                colsample_bylevel=self.colsample_bylevel,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                base_score=self.base_score,
                seed=self.seed
                )
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
        return {'shortname': 'XGB',
                'name': 'XGradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # Parameterized Hyperparameters
        max_depth = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=3))
        learning_rate = cs.add_hyperparameter(UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default=0.1, log=True))
        n_estimators = cs.add_hyperparameter(UniformIntegerHyperparameter
            ("n_estimators", 50, 500, default=100))
        subsample = cs.add_hyperparameter(UniformFloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default=1.0, log=False))
        min_child_weight = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="min_child_weight", lower=1, upper=20, default=1, log=False))

        # Unparameterized Hyperparameters
        max_delta_step = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="max_delta_step", value=0))
        colsample_bytree = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="colsample_bytree", value=1))
        gamma = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="gamma", value=0))
        colsample_bylevel = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="colsample_bylevel", value=1))
        reg_alpha = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="reg_alpha", value=0))
        reg_lambda = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="reg_lambda", value=1))
        base_score = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="base_score", value=0.5))
        scale_pos_weight = cs.add_hyperparameter(UnParametrizedHyperparameter(
            name="scale_pos_weight", value=1))
        return cs
