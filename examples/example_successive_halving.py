"""
==================
Successive Halving
==================
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


def get_smac_object_callback(budget_type):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
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
            )
    return get_smac_object


def main():
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder='/tmp/autosklearn_sh_example_tmp',
        output_folder='/tmp/autosklearn_sh_example_out',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        include_estimators=['extra_trees', 'gradient_boosting', 'random_forest', 'sgd', 'passive_aggressive'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_callback('iterations'),
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder='/tmp/autosklearn_sh_example_tmp',
        output_folder='/tmp/autosklearn_sh_example_out',
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

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1, shuffle=True)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder='/tmp/autosklearn_sh_example_tmp',
        output_folder='/tmp/autosklearn_sh_example_out',
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


if __name__ == '__main__':
    main()
