import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.algorithms import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.pipeline.implementations.util import convert_multioutput_multiclass_to_multilabel
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(AutoSklearnClassificationAlgorithm):

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.criterion = "gini"
        self.splitter = "best"
        self.max_features = 1.0
        self.max_depth = 0.5
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_leaf_nodes = None
        self.min_weight_fraction_leaf = 0.0
        self.class_weight = None

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, SIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        criterion = cs.add_hyperparameter(CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini"))
        splitter = cs.add_hyperparameter(Constant("splitter", "best"))
        max_features = cs.add_hyperparameter(Constant('max_features', 1.0))
        max_depth = cs.add_hyperparameter(UniformFloatHyperparameter(
            'max_depth', 0., 2., default=0.5))
        min_samples_split = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2))
        min_samples_leaf = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1))
        min_weight_fraction_leaf = cs.add_hyperparameter(
            Constant("min_weight_fraction_leaf", 0.0))
        max_leaf_nodes = cs.add_hyperparameter(
            UnParametrizedHyperparameter("max_leaf_nodes", "None"))

        return cs

    def fit(self, X, y, sample_weight=None):

        self.max_features = float(self.max_features)
        if self.max_depth == "None":
            max_depth = self.max_depth = None
        else:
            num_features = X.shape[1]
            self.max_depth = int(self.max_depth)
            max_depth = max(1, int(np.round(self.max_depth * num_features, 0)))
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if self.max_leaf_nodes == "None":
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        probas = super(DecisionTree, self).predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas
