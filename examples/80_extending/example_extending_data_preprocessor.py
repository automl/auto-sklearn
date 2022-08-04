"""
=======================================================
Extending Auto-Sklearn with Data Preprocessor Component
=======================================================

The following example demonstrates how to turn off data preprocessing step in auto-skearn.
"""
from typing import Optional
from pprint import pprint

import autosklearn.classification
import autosklearn.pipeline.components.data_preprocessing
import sklearn.metrics
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


############################################################################
# Create NoPreprocessing component for auto-sklearn
# =================================================
class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()  # Return an empty configuration as there is None


# Add NoPreprocessing component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)

############################################################################
# Create dataset
# ==============

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

############################################################################
# Fit the model without performing data preprocessing
# ===================================================

clf = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    include={"data_preprocessor": ["NoPreprocessing"]},
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={"runcount_limit": 5},
)
clf.fit(X_train, y_train)

# To check that models were found without issue when running examples
assert len(clf.get_models_with_weights()) > 0
print(clf.sprint_statistics())

############################################################################
# Print prediction score and statistics
# =====================================

y_pred = clf.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
pprint(clf.show_models(), indent=4)
