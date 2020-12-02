"""
===================================================================
Restricting the number of hyperparameters for an existing component
===================================================================

The following example demonstrates how to replace an existing
component with a new component, implementing the same classifier,
but with different hyperparameters .
"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.classification \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE


if __name__ == "__main__":
    ############################################################################
    # Subclass auto-sklearn's random forest classifier
    # ================================================

    # This classifier only has one of the hyperparameter's of auto-sklearn's
    # default parametrization (``max_features``). Instead, it also
    # tunes the number of estimators (``n_estimators``).

    class CustomRandomForest(AutoSklearnClassificationAlgorithm):
        def __init__(self,
                     n_estimators,
                     max_features,
                     random_state=None,
                     ):
            self.n_estimators = n_estimators
            self.max_features = max_features
            self.random_state = random_state

        def fit(self, X, y):
            from sklearn.ensemble import RandomForestClassifier

            self.n_estimators = int(self.n_estimators)

            if self.max_features not in ("sqrt", "log2", "auto"):
                max_features = int(X.shape[1] ** float(self.max_features))
            else:
                max_features = self.max_features

            self.estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features=max_features,
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
            return {'shortname': 'RF',
                    'name': 'Random Forest Classifier',
                    'handles_regression': False,
                    'handles_classification': True,
                    'handles_multiclass': True,
                    'handles_multilabel': True,
                    'handles_multioutput': False,
                    'is_deterministic': True,
                    'input': (DENSE, SPARSE, UNSIGNED_DATA),
                    'output': (PREDICTIONS,)}

        @staticmethod
        def get_hyperparameter_search_space(dataset_properties=None):
            cs = ConfigurationSpace()

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
            n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 1000, default_value=100)

            cs.add_hyperparameters([max_features, n_estimators])
            return cs


    # Add custom random forest classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(CustomRandomForest)
    cs = CustomRandomForest.get_hyperparameter_search_space()
    print(cs)

    ############################################################################
    # Data Loading
    # ============

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ############################################################################
    # Fit Random forest classifier to the data
    # ========================================

    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        # Here we exclude auto-sklearn's default random forest component
        exclude_estimators=['random_forest'],
        # Bellow two flags are provided to speed up calculations
        # Not recommended for a real implementation
        initial_configurations_via_metalearning=0,
        smac_scenario_args={'runcount_limit': 1},
    )
    clf.fit(X_train, y_train)

    ############################################################################
    # Print the configuration space
    # =============================

    # Observe that this configuration space only contains our custom random
    # forest, but not auto-sklearn's ``random_forest``
    cs = clf.get_configuration_space(X_train, y_train)
    assert 'random_forest' not in str(cs)
    print(cs)
