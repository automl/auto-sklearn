import numpy as np

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ParamSklearn.components.classification_base import \
    ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, PREDICTIONS
# get our own forests to replace the sklearn ones
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(ParamSklearnClassificationAlgorithm):
    def __init__(self, criterion, max_features, max_depth,
                 min_samples_split, min_samples_leaf,
                 max_leaf_nodes, random_state=None):
        self.criterion = criterion
        self.max_features = float(max_features)

        if max_depth == "None":
            self.max_depth = None
        else:
            self.max_depth = max_depth

        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

        if max_leaf_nodes == "None":
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(max_leaf_nodes)

        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        num_features = X.shape[1]
        max_depth = max(1, self.max_depth * np.ceil(np.log2(num_features)))

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
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
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
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
                'input': (DENSE, ),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini")
        max_features = Constant('max_features', 1.0)
        max_depth = UniformFloatHyperparameter('max_depth', 0., 1., default=1.)
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(criterion)
        cs.add_hyperparameter(max_features)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(max_leaf_nodes)
        return cs
