import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
    IterativeComponentWithSampleWeight,
)
from autosklearn.pipeline.implementations.xgb import \
    CustomXGBRegressor
from autosklearn.pipeline.constants import *


class XGradientBoostingRegressor(
    IterativeComponentWithSampleWeight,
    AutoSklearnRegressionAlgorithm,
):
    def __init__(self,
                 # General Hyperparameters
                 learning_rate, n_estimators, subsample, booster, max_depth,
                 colsample_bylevel, colsample_bytree, reg_alpha, reg_lambda,
                 # Inactive Hyperparameters
                 gamma, min_child_weight, max_delta_step,
                 base_score, scale_pos_weight, n_jobs=1, init=None,
                 random_state=None, verbose=0,
                 # (Conditional) DART Hyperparameters
                 sample_type=None, normalize_type=None, rate_drop=None,
                 ):

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.booster = booster

        booster_args = {
            'sample_type': sample_type,
            'normalize_type': normalize_type,
            'rate_drop': rate_drop,
        }
        if any(v is not None for v in booster_args.values()):
            self.booster_args = booster_args
        else:
            self.booster_args = {}

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
        self.n_jobs = n_jobs

        ## Were there before, didn't touch
        self.init = init
        self.estimator = None

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):

        import sklearn.model_selection
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(
                X, y, random_state=1, test_size=0.1)

        if refit:
            self.estimator = None

        if self.estimator is None:
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
            self.n_jobs = int(self.n_jobs)
            self.base_score = float(self.base_score)
            self.scale_pos_weight = float(self.scale_pos_weight)
            for key in self.booster_args:
                try:
                    self.booster_args[key] = float(self.booster_args[key])
                except:
                    pass

            self.objective = 'reg:linear'
            self.eval_metric = 'mae'

            arguments = dict(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=n_iter,
                silent=self.silent,
                booster=self.booster,
                objective=self.objective,
                n_jobs=self.n_jobs,
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
                random_state=self.seed,
                **self.booster_args
            )

            self.estimator = CustomXGBRegressor(
                tree_method='auto',
                **arguments
            )
            self.estimator.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=self.eval_metric,
                verbose=False,
            )

        elif not self.configuration_fully_fitted():
            n_estimators_before_fit = self.estimator.n_estimators
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)
            self.estimator.fit(
                X_train, y_train,
                xgb_model=self.estimator.get_booster(),
                eval_set=[(X_train, y_train), (X_test, y_test)],
                early_stopping_rounds=int(self.n_estimators / 10),
                eval_metric=self.eval_metric,
                verbose=False,
            )
            if (
                        n_estimators_before_fit + n_iter
                    > self.estimator.best_iteration + int(
                        self.n_estimators / 10)
            ) or (
                    self.estimator.n_estimators >= self.n_estimators
            ):
                self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'XGB',
            'name': 'XGradient Boosting Regressor',
            'handles_regression': True,
            'handles_classification': False,
            'handles_multiclass': False,
            'handles_multilabel': False,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (PREDICTIONS,)
            }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # Parameterized Hyperparameters
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=3
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1,
            log=True
        )
        n_estimators = Constant("n_estimators", 512)
        booster = CategoricalHyperparameter(
            "booster", ["gbtree", "dart"]
        )
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default_value=1.0,
            log=False
        )
        min_child_weight = UniformIntegerHyperparameter(
            name="min_child_weight", lower=1e-10,
            upper=20, default_value=1, log=False
        )
        colsample_bytree = UniformFloatHyperparameter(
            name="colsample_bytree", lower=0.1, upper=1.0, default_value=1,
        )
        colsample_bylevel = UniformFloatHyperparameter(
            name="colsample_bylevel", lower=0.1, upper=1.0, default_value=1,
        )
        reg_alpha = UniformFloatHyperparameter(
            name="reg_alpha", lower=1e-10, upper=1e-1, log=True,
            default_value=1e-10)
        reg_lambda = UniformFloatHyperparameter(
            name="reg_lambda", lower=1e-10, upper=1e-1, log=True,
            default_value=1e-10)

        # DART Hyperparameters
        sample_type = CategoricalHyperparameter(
            'sample_type', ['uniform', 'weighted'], default_value='uniform',
        )
        normalize_type = CategoricalHyperparameter(
            'normalize_type', ['tree', 'forest'], default_value='tree',
        )
        rate_drop = UniformFloatHyperparameter(
            'rate_drop', 1e-10, 1 - (1e-10), default_value=0.5,
        )

        # Unparameterized Hyperparameters
        # https://xgboost.readthedocs.io/en/latest//parameter.html
        # minimum loss reduction required to make a further partition on a
        # leaf node of the tree
        gamma = UnParametrizedHyperparameter(
            name="gamma", value=0)
        # absolute regularization (in contrast to eta), comparable to
        # gradient clipping in deep learning - according to the internet this
        #  is most important for unbalanced data
        max_delta_step = UnParametrizedHyperparameter(
            name="max_delta_step", value=0)
        base_score = UnParametrizedHyperparameter(
            name="base_score", value=0.5)
        scale_pos_weight = UnParametrizedHyperparameter(
            name="scale_pos_weight", value=1)

        cs.add_hyperparameters([
            # Active
            max_depth, learning_rate, n_estimators, booster,
            subsample, colsample_bytree, colsample_bylevel,
            reg_alpha, reg_lambda,
            # DART
            sample_type, normalize_type, rate_drop,
            # Inactive
            min_child_weight, max_delta_step, gamma,
            base_score, scale_pos_weight
        ])

        sample_type_condition = EqualsCondition(
            sample_type, booster, 'dart',
        )
        normalize_type_condition = EqualsCondition(
            normalize_type, booster, 'dart',
        )
        rate_drop_condition = EqualsCondition(
            rate_drop, booster, 'dart',
        )

        cs.add_conditions([
            sample_type_condition, normalize_type_condition,
            rate_drop_condition,
        ])
        return cs
