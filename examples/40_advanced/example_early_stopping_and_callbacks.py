"""
============================
Early stopping and Callbacks
============================

The example below shows how we can use the ``get_trials_callback`` parameter of
auto-sklearn to implement an early-stopping mechanism through a callback.

These callbacks give access to the result of each model + hyperparameter configuration
optimized by SMAC, the underlying optimizer for autosklearn. By checking the cost of
a result, we can implement a simple yet effective early stopping mechanism!

Do note however, this does not provide any access to the ensembles that autosklearn
produces, only the individual models. You may wish to perform a more sophisticated
early stopping mechanism such that there are enough good models for autosklearn to build
and ensemble with. This is here to provide a simple example.
"""
from __future__ import annotations

from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue


############################################################################
# Build and fit a classifier
# ==========================
def callback(
    smbo: SMBO,
    run_info: RunInfo,
    result: RunValue,
    time_left: float,
) -> bool | None:
    """Stop early if we get a very low cost value for a single run

    The return value indicates to SMAC whether to stop or not. False will
    stop the search process while any other value will mean it continues.
    """
    # You can find out the parameters in the SMAC documentation
    # https://automl.github.io/SMAC3/main/
    if result.cost <= 0.02:
        print("Stopping!")
        print(run_info)
        print(result)
        return False


X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120, per_run_time_limit=30, get_trials_callback=callback
)
automl.fit(X_train, y_train, dataset_name="breast_cancer")

############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
