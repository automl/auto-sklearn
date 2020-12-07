"""
====================================================
Extending Auto-Sklearn with Classification Component
====================================================

The following example demonstrates how to create a new classification
component for using in auto-sklearn.
"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.base \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, \
    PREDICTIONS

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


############################################################################
# Create MLP classifier component for auto-sklearn
# ================================================

class MLPClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 hidden_layer_depth,
                 num_nodes_per_layer,
                 activation,
                 alpha,
                 solver,
                 random_state=None,
                 ):
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y):
        self.num_nodes_per_layer = int(self.num_nodes_per_layer)
        self.hidden_layer_depth = int(self.hidden_layer_depth)
        self.alpha = float(self.alpha)

        from sklearn.neural_network import MLPClassifier
        hidden_layer_sizes = tuple(self.num_nodes_per_layer for i in range(self.hidden_layer_depth))

        self.estimator = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                       activation=self.activation,
                                       alpha=self.alpha,
                                       solver=self.solver,
                                       random_state=self.random_state,
                                       )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MLP Classifier',
                'name': 'MLP CLassifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': False,
                # Both input and output must be tuple(iterable)
                'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                'output': [PREDICTIONS]
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        hidden_layer_depth = UniformIntegerHyperparameter(
            name="hidden_layer_depth", lower=1, upper=3, default_value=1
        )
        num_nodes_per_layer = UniformIntegerHyperparameter(
            name="num_nodes_per_layer", lower=16, upper=216, default_value=32
        )
        activation = CategoricalHyperparameter(
            name="activation", choices=['identity', 'logistic', 'tanh', 'relu'],
            default_value='relu'
        )
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0001, upper=1.0, default_value=0.0001
        )
        solver = CategoricalHyperparameter(
            name="solver", choices=['lbfgs', 'sgd', 'adam'], default_value='adam'
        )
        cs.add_hyperparameters([hidden_layer_depth,
                                num_nodes_per_layer,
                                activation,
                                alpha,
                                solver,
                                ])
        return cs


# Add MLP classifier component to auto-sklearn.
autosklearn.pipeline.components.classification.add_classifier(MLPClassifier)
cs = MLPClassifier.get_hyperparameter_search_space()
print(cs)


if __name__ == "__main__":

    ############################################################################
    # Data Loading
    # ============

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ############################################################################
    # Fit MLP classifier to the data
    # ==============================

    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        include_estimators=['MLPClassifier'],
        # Bellow two flags are provided to speed up calculations
        # Not recommended for a real implementation
        initial_configurations_via_metalearning=0,
        smac_scenario_args={'runcount_limit': 5},
    )
    clf.fit(X_train, y_train)

    ############################################################################
    # Print test accuracy and statistics
    # ==================================

    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.show_models())
