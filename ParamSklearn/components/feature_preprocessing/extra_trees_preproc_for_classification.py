import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ParamSklearn.components.base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.constants import *


class ExtraTreesPreprocessor(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features,
                 max_leaf_nodes_or_max_depth="max_depth",
                 bootstrap=False, max_leaf_nodes=None, max_depth="None",
                 min_weight_fraction_leaf=0.0,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None):

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
                # if use_max_depth == "True":
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
        self.class_weight = class_weight
        self.preprocessor = None

    def fit(self, X, Y, sample_weight=None):
        num_features = X.shape[1]
        max_features = int(
            float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        self.preprocessor = ExtraTreesClassifier(
            n_estimators=0, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
            max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state, class_weight=self.class_weight,
            warm_start=True
        )
        # JTS TODO: I think we might have to copy here if we want self.estimator
        # to always be consistent on sigabort
        while len(self.preprocessor.estimators_) < self.n_estimators:
            tmp = self.preprocessor  # TODO copy ?
            tmp.n_estimators += self.estimator_increment
            tmp.fit(X, Y, sample_weight=sample_weight)
            self.preprocessor = tmp
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier Preprocessing',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

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
