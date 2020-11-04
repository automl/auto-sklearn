#!/usr/bin/env python3

import os
import argparse
import numpy as np
import openml

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy
from remove_dataset_from_metadata import remove_dataset
import score_ensemble


def load_task(task_id):
    """Function used for loading data."""
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    dataset = openml.datasets.get_dataset(task.dataset_id)
    _, _, cat = dataset.get_data(return_categorical_indicator=True,
                                 target=task.target_name)
    del _
    del dataset
    cat = ['categorical' if c else 'numerical' for c in cat]

    unique = np.unique(y_train)
    mapping = {unique_value: i for i, unique_value in enumerate(unique)}
    y_train = np.array([mapping[value] for value in y_train])
    y_test = np.array([mapping[value] for value in y_test])

    return X_train, y_train, X_test, y_test, cat


def run_experiment(working_directory,
                   time_limit,
                   per_run_time_limit,
                   task_id,
                   seed,
                   use_metalearning,
                   ):
    # set this to local dataset cache
    # openml.config.cache_directory = os.path.join(working_directory, "../cache")

    seed_dir = os.path.join(working_directory, str(seed))
    try:
        os.makedirs(seed_dir)
    except Exception:
        print("Directory {0} aleardy created.".format(seed_dir))

    tmp_dir = os.path.join(seed_dir, str(task_id))

    # With metalearning
    if use_metalearning is True:
        # path to the original metadata directory.
        metadata_directory = os.path.abspath(os.path.dirname(__file__))
        metadata_directory = os.path.join(metadata_directory,
                                          "../../../autosklearn/metalearning/files/")

        # Create new metadata directory not containing task_id.
        new_metadata_directory = os.path.abspath(os.path.join(working_directory,
                                                              "metadata_%i" % task_id))

        try:
            os.makedirs(new_metadata_directory)
        except OSError:
            pass  # pass because new metadata is created for this task.

        # remove the given task id from metadata directory.
        remove_dataset(metadata_directory, new_metadata_directory, task_id)

        automl_arguments = {
            'time_left_for_this_task': time_limit,
            'per_run_time_limit': per_run_time_limit,
            'initial_configurations_via_metalearning': 25,
            'ensemble_size': 0,
            'seed': seed,
            'memory_limit': 3072,
            'resampling_strategy': 'holdout',
            'resampling_strategy_arguments': {'train_size': 0.67},
            'tmp_folder': tmp_dir,
            'delete_tmp_folder_after_terminate': False,
            'disable_evaluator_output': False,
            'metadata_directory': new_metadata_directory
        }

    # Without metalearning
    else:
        automl_arguments = {
            'time_left_for_this_task': time_limit,
            'per_run_time_limit': per_run_time_limit,
            'initial_configurations_via_metalearning': 0,
            'ensemble_size': 0,
            'seed': seed,
            'memory_limit': 3072,
            'resampling_strategy': 'holdout',
            'resampling_strategy_arguments': {'train_size': 0.67},
            'tmp_folder': tmp_dir,
            'delete_tmp_folder_after_terminate': False,
            'disable_evaluator_output': False,
        }

    automl = AutoSklearnClassifier(**automl_arguments)

    X_train, y_train, X_test, y_test, cat = load_task(task_id)

    automl.fit(X_train, y_train,
               dataset_name=str(task_id),
               X_test=X_test, y_test=y_test,
               metric=balanced_accuracy)


def main(working_directory,
         output_file,
         task_id,
         seed,
         model,
         time_limit,
         per_run_time_limit):
    # vanilla and metalearning must be called first before ensemble and
    # meta_ensemble can be called, respectively.
    if model == "vanilla":
        run_experiment(working_directory,
                       time_limit,
                       per_run_time_limit,
                       task_id,
                       seed,
                       use_metalearning=False,
                       )
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=1,
                            )
    elif model == "metalearning":
        run_experiment(working_directory,
                       time_limit,
                       per_run_time_limit,
                       task_id,
                       seed,
                       use_metalearning=True,
                       )
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=1,
                            )
    else:
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=50,
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-directory', type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--time-limit", type=int, required=True)
    parser.add_argument("--per-runtime-limit", type=int, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()
    working_directory = args.working_directory  # logdir/vanilla or logdir/metalearning
    output_file = args.output_file
    task_id = args.task_id
    seed = args.seed
    model = args.model
    time_limit = args.time_limit
    per_run_time_limit = args.per_runtime_limit

    main(working_directory,
         output_file,
         task_id,
         seed,
         model,
         time_limit,
         per_run_time_limit,
         )
