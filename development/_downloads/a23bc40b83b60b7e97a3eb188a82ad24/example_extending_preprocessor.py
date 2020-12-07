"""
==================================================
Extending Auto-Sklearn with Preprocessor Component
==================================================

The following example demonstrates how to create a wrapper around the linear
discriminant analysis (LDA) algorithm from sklearn and use it as a preprocessor
in auto-sklearn.
"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base \
    import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA
from autosklearn.util.common import check_none

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


############################################################################
# Create LDA component for auto-sklearn
# =====================================
class LDA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, solver, tol, shrinkage=None, random_state=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.tol = tol
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y=None):
        if check_none(self.shrinkage):
            self.shrinkage = None
        else:
            self.shrinkage = float(self.shrinkage)
        self.tol = float(self.tol)

        import sklearn.discriminant_analysis
        self.preprocessor = \
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                shrinkage=self.shrinkage,
                solver=self.solver,
                tol=self.tol,
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        solver = CategoricalHyperparameter(
            name="solver", choices=['svd', 'lsqr', 'eigen'], default_value='svd'
        )
        shrinkage = UniformFloatHyperparameter(
            name="shrinkage", lower=0.0, upper=1.0, default_value=0.5
        )
        tol = UniformFloatHyperparameter(
            name="tol", lower=0.0001, upper=1, default_value=0.0001
        )
        cs.add_hyperparameters([solver, shrinkage, tol])
        shrinkage_condition = InCondition(shrinkage, solver, ['lsqr', 'eigen'])
        cs.add_condition(shrinkage_condition)
        return cs


# Add LDA component to auto-sklearn.
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(LDA)


if __name__ == "__main__":

    ############################################################################
    # Create dataset
    # ==============

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ############################################################################
    # Configuration space
    # ===================

    cs = LDA.get_hyperparameter_search_space()
    print(cs)

    ############################################################################
    # Fit the model using LDA as preprocessor
    # =======================================

    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        include_preprocessors=['LDA'],
        # Bellow two flags are provided to speed up calculations
        # Not recommended for a real implementation
        initial_configurations_via_metalearning=0,
        smac_scenario_args={'runcount_limit': 5},
    )
    clf.fit(X_train, y_train)

    ############################################################################
    # Print prediction score and statistics
    # =====================================

    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.show_models())
