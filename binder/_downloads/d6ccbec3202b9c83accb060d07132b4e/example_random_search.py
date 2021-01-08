"""
=============
Random Search
=============

A crucial feature of *auto-sklearn* is automatically optimizing the hyperparameters through SMAC,
introduced `here <http://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf>`_. Additionally, it
is possible to use `random search <http://www.jmlr.org/papers/v13/bergstra12a.html>`_ instead of
SMAC, as demonstrated in the example below. Furthermore, the example also demonstrates how to use
`Random Online Aggressive Racing (ROAR) <http://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf>`_
as yet another alternative optimizatino strategy.
Both examples are intended to show how the optimization strategy in *auto-sklearn* can be adapted.
"""  # noqa (links are too long)

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

import autosklearn.classification


if __name__ == "__main__":
    ############################################################################
    # Data Loading
    # ============

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)


    ############################################################################
    # Fit a classifier using ROAR
    # ===========================
    def get_roar_object_callback(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        metalearning_configurations,
        n_jobs,
        dask_client,
    ):
        """Random online adaptive racing."""

        if n_jobs > 1 or (dask_client and len(dask_client.nthreads()) > 1):
            raise ValueError("Please make sure to guard the code invoking Auto-sklearn by "
                             "`if __name__ == '__main__'` and remove this exception.")

        scenario = Scenario(scenario_dict)
        return ROAR(
            scenario=scenario,
            rng=seed,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            run_id=seed,
            dask_client=dask_client,
            n_jobs=n_jobs,
        )


    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60, per_run_time_limit=15,
        tmp_folder='/tmp/autosklearn_random_search_example_tmp',
        output_folder='/tmp/autosklearn_random_search_example_out',
        get_smac_object_callback=get_roar_object_callback,
        initial_configurations_via_metalearning=0,
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    print('#' * 80)
    print('Results for ROAR.')
    # Print the final ensemble constructed by auto-sklearn via ROAR.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


    ############################################################################
    # Fit a classifier using Random Search
    # ====================================
    def get_random_search_object_callback(
            scenario_dict,
            seed,
            ta,
            ta_kwargs,
            metalearning_configurations,
            n_jobs,
            dask_client
    ):
        """Random search."""

        if n_jobs > 1 or (dask_client and len(dask_client.nthreads()) > 1):
            raise ValueError("Please make sure to guard the code invoking Auto-sklearn by "
                             "`if __name__ == '__main__'` and remove this exception.")

        scenario_dict['minR'] = len(scenario_dict['instances'])
        scenario_dict['initial_incumbent'] = 'RANDOM'
        scenario = Scenario(scenario_dict)
        return ROAR(
            scenario=scenario,
            rng=seed,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            run_id=seed,
            dask_client=dask_client,
            n_jobs=n_jobs,
        )


    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=15,
        tmp_folder='/tmp/autosklearn_random_search_example_tmp',
        output_folder='/tmp/autosklearn_random_search_example_out',
        get_smac_object_callback=get_random_search_object_callback,
        initial_configurations_via_metalearning=0,
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    print('#' * 80)
    print('Results for random search.')

    # Print the final ensemble constructed by auto-sklearn via random search.
    print(automl.show_models())

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())

    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
