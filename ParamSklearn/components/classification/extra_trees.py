import numpy as np

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, PREDICTIONS

# get our own forests to replace the sklearn ones
from ParamSklearn.implementations import forest


class ExtraTreesClassifier(ParamSklearnClassificationAlgorithm):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split,  max_features, max_leaf_nodes_or_max_depth="max_depth", #use_max_depth=False,
                 bootstrap=False, max_leaf_nodes=None, max_depth="None",
                 oob_score=False, n_jobs=1, random_state=None, verbose=0):

        self.n_estimators = int(n_estimators)
        self.estimator_increment = 10
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion

        if max_leaf_nodes_or_max_depth == "max_depth":
            self.max_leaf_nodes = None
            if max_depth == "None":
                self.max_depth = None
            else:
                self.max_depth = int(max_depth)
            #if use_max_depth == "True":
            #    self.max_depth = int(max_depth)
            #elif use_max_depth == "False":
            #    self.max_depth = None
        else:
            if max_leaf_nodes == "None":
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(max_leaf_nodes)
            self.max_depth = None

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)

        self.max_features = float(max_features)

        if bootstrap == "True":
            self.bootstrap = True
        elif bootstrap == "False":
            self.bootstrap = False

        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, Y):
        num_features = X.shape[1]
        max_features = int(float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        self.estimator = forest.ExtraTreesClassifier(
            n_estimators=0, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
            max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state,
            warm_start = True
        )
        # JTS TODO: I think we might have to copy here if we want self.estimator
        #           to always be consistent on sigabort
        while len(self.estimator.estimators_) < self.n_estimators:
            tmp = self.estimator # TODO copy ?
            tmp.n_estimators += self.estimator_increment
            tmp.fit(X, Y)
            self.estimator = tmp
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
    def get_properties():
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
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
                'input': (DENSE, ),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        #use_max_depth = CategoricalHyperparameter(
        #    name="use_max_depth", choices=("True", "False"), default="False")
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default="False")

        # Copied from random_forest.py
        #n_estimators = UniformIntegerHyperparameter(
        #    "n_estimators", 10, 100, default=10)
        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini")
        #max_features = UniformFloatHyperparameter(
        #    "max_features", 0.01, 0.5, default=0.1)
        max_features = UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1)
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1)

        # Unparametrized
        #max_leaf_nodes_or_max_depth = UnParametrizedHyperparameter(
        #    name="max_leaf_nodes_or_max_depth", value="max_depth")
            # CategoricalHyperparameter("max_leaf_nodes_or_max_depth",
            # choices=["max_leaf_nodes", "max_depth"], default="max_depth")
        #max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes",
        #                                              value="None")
            # UniformIntegerHyperparameter(
            # name="max_leaf_nodes", lower=10, upper=1000, default=)

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(criterion)
        cs.add_hyperparameter(max_features)
        #cs.add_hyperparameter(use_max_depth)
        cs.add_hyperparameter(max_depth)
        #cs.add_hyperparameter(max_leaf_nodes_or_max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        #cs.add_hyperparameter(max_leaf_nodes)
        cs.add_hyperparameter(bootstrap)

        # Conditions
        # Not applicable because max_leaf_nodes is no legal value of the parent
        #cond_max_leaf_nodes_or_max_depth = \
        #    EqualsCondition(child=max_leaf_nodes,
        #                    parent=max_leaf_nodes_or_max_depth,
        #                    value="max_leaf_nodes")
        #cond2_max_leaf_nodes_or_max_depth = \
        #    EqualsCondition(child=use_max_depth,
        #                    parent=max_leaf_nodes_or_max_depth,
        #                    value="max_depth")

        #cond_max_depth = EqualsCondition(child=max_depth, parent=use_max_depth,
                                         #value="True")
        #cs.add_condition(cond_max_leaf_nodes_or_max_depth)
        #cs.add_condition(cond2_max_leaf_nodes_or_max_depth)
        #cs.add_condition(cond_max_depth)

        return cs
