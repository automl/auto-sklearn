"""
==================
Successive Halving
==================

This advanced  example illustrates how to interact with
the SMAC callback and get relevant information from the run, like
the number of iterations. Particularly, it exemplifies how to select
the intensification strategy to use in smac, in this case:
`SuccessiveHalving <http://proceedings.mlr.press/v80/falkner18a/falkner18a-supp.pdf>`_.
"""  # noqa (links are too long)


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


if __name__ == "__main__":
    ############################################################################
    # Define a callback that instantiates SuccessiveHalving
    # =====================================================

    def get_smac_object_callback(budget_type):
        def get_smac_object(
            scenario_dict,
            seed,
            ta,
            ta_kwargs,
            metalearning_configurations,
            n_jobs,
            dask_client,
        ):
            from smac.facade.smac_ac_facade import SMAC4AC
            from smac.intensification.successive_halving import SuccessiveHalving
            from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
            from smac.scenario.scenario import Scenario

            if n_jobs > 1 or (dask_client and len(dask_client.nthreads()) > 1):
                raise ValueError("Please make sure to guard the code invoking Auto-sklearn by "
                                 "`if __name__ == '__main__'` and remove this exception.")

            scenario = Scenario(scenario_dict)
            if len(metalearning_configurations) > 0:
                default_config = scenario.cs.get_default_configuration()
                initial_configurations = [default_config] + metalearning_configurations
            else:
                initial_configurations = None
            rh2EPM = RunHistory2EPM4LogCost

            ta_kwargs['budget_type'] = budget_type

            return SMAC4AC(
                scenario=scenario,
                rng=seed,
                runhistory2epm=rh2EPM,
                tae_runner=ta,
                tae_runner_kwargs=ta_kwargs,
                initial_configurations=initial_configurations,
                run_id=seed,
                intensifier=SuccessiveHalving,
                intensifier_kwargs={
                    'initial_budget': 10.0,
                    'max_budget': 100,
                    'eta': 2,
                    'min_chall': 1},
                n_jobs=n_jobs,
                dask_client=dask_client,
            )
        return get_smac_object


    ############################################################################
    # Data Loading
    # ============

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    ############################################################################
    # Build and fit a classifier
    # ==========================

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=40,
        per_run_time_limit=10,
        tmp_folder='/tmp/autosklearn_sh_example_tmp',
        output_folder='/tmp/autosklearn_sh_example_out',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        include_estimators=['extra_trees', 'gradient_boosting', 'random_forest', 'sgd',
                            'passive_aggressive'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_callback('iterations'),
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # We can also use cross-validation with successive halving
    # ========================================================

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=40,
        per_run_time_limit=10,
        tmp_folder='/tmp/autosklearn_sh_example_tmp_01',
        output_folder='/tmp/autosklearn_sh_example_out_01',
        disable_evaluator_output=False,
        resampling_strategy='cv',
        include_estimators=['extra_trees', 'gradient_boosting', 'random_forest', 'sgd',
                            'passive_aggressive'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_callback('iterations'),
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    automl.refit(X_train, y_train)
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # Use an iterative fit cross-validation with successive halving
    # =============================================================

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=40,
        per_run_time_limit=10,
        tmp_folder='/tmp/autosklearn_sh_example_tmp_cv_02',
        output_folder='/tmp/autosklearn_sh_example_out_cv_02',
        disable_evaluator_output=False,
        resampling_strategy='cv-iterative-fit',
        include_estimators=['extra_trees', 'gradient_boosting', 'random_forest', 'sgd',
                            'passive_aggressive'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_callback('iterations'),
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    automl.refit(X_train, y_train)
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # Next, we see the use of subsampling as a budget in Auto-sklearn
    # ===============================================================

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=40,
        per_run_time_limit=10,
        tmp_folder='/tmp/autosklearn_sh_example_tmp_03',
        output_folder='/tmp/autosklearn_sh_example_out_03',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        get_smac_object_callback=get_smac_object_callback('subsample'),
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # Mixed budget approach
    # =====================
    # Finally, there's a mixed budget type which uses iterations where possible and
    # subsamples otherwise

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=40,
        per_run_time_limit=10,
        tmp_folder='/tmp/autosklearn_sh_example_tmp_04',
        output_folder='/tmp/autosklearn_sh_example_out_04',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        include_estimators=['extra_trees', 'gradient_boosting', 'random_forest', 'sgd'],
        get_smac_object_callback=get_smac_object_callback('mixed'),
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
