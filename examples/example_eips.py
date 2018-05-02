"""
====
EIPS
====

This example demonstrates the usage of a different acquisition function inside SMAC, namely
`Expected Improvement per Second (EIPS) <https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.html>_`.
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.utils.util_funcs import get_types
from smac.facade.smac_facade import SMAC
from smac.optimizer.acquisition import EIPS
from smac.runhistory.runhistory2epm import RunHistory2EPM4EIPS
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType

import autosklearn.classification


def get_eips_object_callback(
        scenario_dict,
        seed,
        ta,
        backend,
        metalearning_configurations,
        runhistory,
        run_id,
):
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario = Scenario(scenario_dict)
    rh2EPM = RunHistory2EPM4EIPS(
        num_params=len(scenario.cs.get_hyperparameters()),
        scenario=scenario,
        success_states=[
            StatusType.SUCCESS,
            StatusType.MEMOUT,
            StatusType.TIMEOUT,
            StatusType.CRASHED
        ],
        impute_censored_data=False,
        impute_state=None
    )
    types, bounds = get_types(scenario.cs,
                              scenario.feature_array)
    model = UncorrelatedMultiObjectiveRandomForestWithInstances(
        ['cost', 'runtime'],
        types=types,
        bounds=bounds,
        instance_features=scenario.feature_array,
        rf_kwargs={'seed': 1,},
    )
    acquisition_function = EIPS(model)
    return SMAC(
        runhistory=runhistory,
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        runhistory2epm=rh2EPM,
        model=model,
        acquisition_function=acquisition_function,
        run_id=run_id,
    )


def main():
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_eips_example_tmp',
        output_folder='/tmp/autosklearn_eips_example_out',
        get_smac_object_callback=get_eips_object_callback,
        initial_configurations_via_metalearning=0,
    )
    automl.fit(X_train, y_train, dataset_name='digits')

    # Print the final ensemble constructed by auto-sklearn via ROAR.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))



if __name__ == '__main__':
    main()
