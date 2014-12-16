import numpy as np
import sklearn.ensemble

from HPOlibConfigSpace.conditions import EqualsCondition, OrConjunction

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from ..classification_base import AutoSklearnClassificationAlgorithm

"""
            param_dist = {"max_features": numpy.linspace(0.1, 1, num=10),
                          "learning_rate": 2**numpy.linspace(-1, -10, num=10),
                          "max_depth": range(1, 11),
                          "min_samples_leaf": range(2, 20, 2),
                          "n_estimators": range(10, 110, 10)}
            param_list = [{"max_features": max_features,
                           "learning_rate": learning_rate,
                           "max_depth": max_depth,
                           "min_samples_leaf": min_samples_leaf,
                           "n_estimators": n_estimators}]
            param_list.extend(list(ParameterSampler(param_dist, n_iter=random_iter-1, random_state
"""


class GradientBoostingClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf, max_features,
                 max_leaf_nodes_or_max_depth, max_depth,
                 max_leaf_nodes=None, loss='deviance',
                 warm_start=False, init=None, random_state=None, verbose=0):
        """
        Parameters
        ----------
        loss : {'deviance'}, optional (default='deviance')
            loss function to be optimized. 'deviance' refers to
            deviance (= logistic regression) for classification
            with probabilistic outputs.

        learning_rate : float, optional (default=0.1)
            learning rate shrinks the contribution of each tree by `learning_rate`.
            There is a trade-off between learning_rate and n_estimators.

        n_estimators : int (default=100)
            The number of boosting stages to perform. Gradient boosting
            is fairly robust to over-fitting so a large number usually
            results in better performance.

        max_depth : integer, optional (default=3)
            maximum depth of the individual regression estimators. The maximum
            depth limits the number of nodes in the tree. Tune this parameter
            for best performance; the best value depends on the interaction
            of the input variables.
            Ignored if ``max_samples_leaf`` is not None.

        min_samples_split : integer, optional (default=2)
            The minimum number of samples required to split an internal node.

        min_samples_leaf : integer, optional (default=1)
            The minimum number of samples required to be at a leaf node.

        subsample : float, optional (default=1.0)
            The fraction of samples to be used for fitting the individual base
            learners. If smaller than 1.0 this results in Stochastic Gradient
            Boosting. `subsample` interacts with the parameter `n_estimators`.
            Choosing `subsample < 1.0` leads to a reduction of variance
            and an increase in bias.

        max_features : int, float, string or None, optional (default="auto")
            The number of features to consider when looking for the best split:
              - If int, then consider `max_features` features at each split.
              - If float, then `max_features` is a percentage and
                `int(max_features * n_features)` features are considered at each
                split.
              - If "auto", then `max_features=sqrt(n_features)`.
              - If "sqrt", then `max_features=sqrt(n_features)`.
              - If "log2", then `max_features=log2(n_features)`.
              - If None, then `max_features=n_features`.

            Choosing `max_features < n_features` leads to a reduction of variance
            and an increase in bias.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
            If not None then ``max_depth`` will be ignored.

        init : BaseEstimator, None, optional (default=None)
            An estimator object that is used to compute the initial
            predictions. ``init`` has to provide ``fit`` and ``predict``.
            If None it uses ``loss.init_estimator``.

        verbose : int, default: 0
            Enable verbose output. If 1 then it prints progress and performance
            once in a while (the more trees the lower the frequency). If greater
            than 1 then it prints progress and performance for every tree.

        warm_start : bool, default: False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just erase the
            previous solution.
        """
        self.max_leaf_nodes_or_max_depth = str(max_leaf_nodes_or_max_depth)

        if self.max_leaf_nodes_or_max_depth == "max_depth":
            self.max_depth = int(max_depth)
            self.max_leaf_nodes = None
        elif self.max_leaf_nodes_or_max_depth == "max_leaf_nodes":
            self.max_depth = None
            self.max_leaf_nodes = int(max_leaf_nodes)
        else:
            raise ValueError("max_leaf_nodes_or_max_depth sould be in "
                             "('max_leaf_nodes', 'max_depth'): %s" %
                             self.max_leaf_nodes_or_max_depth)

        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        if max_features in ("sqrt", "log2", "auto"):
            raise ValueError("'max_features' should be a float: %s" % max_features)
        self.max_features = float(max_features)
        if self.max_features > 1:
            raise ValueError("'max features' in should be < 1, you set %f" %
                             self.max_features)
        self.loss = loss
        self.warm_start = bool(warm_start)
        self.init = init
        self.random_state = random_state
        self.verbose = int(verbose)

    def fit(self, X, Y):
        self.estimator = sklearn.ensemble.GradientBoostingClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            loss=self.loss,
            max_depth=self.max_depth,
            warm_start=self.warm_start,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose
        )
        return self.estimator.fit(X, Y)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def scores(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_meta_information():
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': False,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space():
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.0001, upper=1, default=0.1, log=True)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.1, upper=2, default=1.0, log=False)

        # Unparametrized
        max_leaf_nodes_or_max_depth = UnParametrizedHyperparameter(
            name="max_leaf_nodes_or_max_depth", value="max_depth")
            # CategoricalHyperparameter(
            # "max_leaf_nodes_or_max_depth",
            # choices=["max_leaf_nodes", "max_depth"], default="max_depth")
        max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes",
                                                      value="None")
            # UniformIntegerHyperparameter(
            # name="max_leaf_nodes", lower=10, upper=1000, default=)

        # Copied from random_forest.py
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=10, upper=1000, default=10, log=False)
        max_features = UniformFloatHyperparameter(
            name="max_features", lower=0.01, upper=1.0, default=1.0)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=3, log=False)
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=1, upper=20, default=2, log=False)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default=1, log=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(learning_rate)
        cs.add_hyperparameter(max_features)
        cs.add_hyperparameter(max_leaf_nodes_or_max_depth)
        cs.add_hyperparameter(max_leaf_nodes)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(subsample)

        # Conditions
        cond_max_leaf_nodes_or_max_depth = \
            EqualsCondition(child=max_leaf_nodes,
                            parent=max_leaf_nodes_or_max_depth,
                            value="max_leaf_nodes")

        cs.add_condition(cond_max_leaf_nodes_or_max_depth)

        return cs

    def __str__(self):
        return "AutoSklearn GradientBoosting Classifier"
