import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none


class ExtraTreesPreprocessorClassification(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None):

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.preprocessor = None

    def fit(self, X, Y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        self.n_estimators = int(self.n_estimators)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)

        self.bootstrap = check_for_bool(self.bootstrap)
        self.n_jobs = int(self.n_jobs)
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        self.max_features = self.max_features
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.verbose = int(self.verbose)

        max_features = int(X.shape[1] ** float(self.max_features))
        estimator = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            max_features=max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            class_weight=self.class_weight)
        estimator.fit(X, Y, sample_weight=sample_weight)
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
        return {'shortname': 'ETC',
                'name': 'Extra Trees Classifier Preprocessing',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")
        max_features = UniformFloatHyperparameter("max_features", 0, 1,
                                                  default_value=0.5)

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            'min_weight_fraction_leaf', 0.)
        min_impurity_decrease = UnParametrizedHyperparameter(
            'min_impurity_decrease', 0.)

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False")

        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, max_leaf_nodes, min_samples_split,
                                min_samples_leaf, min_weight_fraction_leaf,
                                min_impurity_decrease, bootstrap])

        return cs
