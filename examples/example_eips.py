"""
====
EIPS
====

This example demonstrates the usage of a different acquisition function inside SMAC, namely
`Expected Improvement per Second (EIPS) <https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.html>`_.
"""  # noqa (links are too long)

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.epm.util_funcs import get_types
from smac.facade.smac_ac_facade import SMAC4AC
from smac.optimizer.acquisition import EIPS
from smac.runhistory.runhistory2epm import RunHistory2EPM4EIPS
from smac.scenario.scenario import Scenario

import autosklearn.classification


############################################################################
# EIPS callback
# =============
# create a callack to change the acquisition function inside SMAC
def get_eips_object_callback(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
):
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario = Scenario(scenario_dict)
    types, bounds = get_types(scenario.cs,
                              scenario.feature_array)
    model_kwargs = dict(
        target_names=['cost', 'runtime'],
        types=types,
        bounds=bounds,
        instance_features=scenario.feature_array,
        rf_kwargs={'seed': 1, },
    )
    return SMAC4AC(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        runhistory2epm=RunHistory2EPM4EIPS,
        runhistory2epm_kwargs={},
        model=UncorrelatedMultiObjectiveRandomForestWithInstances,
        model_kwargs=model_kwargs,
        acquisition_function=EIPS,
        run_id=seed,
    )


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Building and fitting the classifier
# ===================================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_eips_example_tmp',
    output_folder='/tmp/autosklearn_eips_example_out',
    get_smac_object_callback=get_eips_object_callback,
    initial_configurations_via_metalearning=0,
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

# Print the final ensemble constructed by auto-sklearn via ROAR.
print(automl.show_models())

############################################################################
# Print statistics about the auto-sklearn run
# ===========================================

# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())

############################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
