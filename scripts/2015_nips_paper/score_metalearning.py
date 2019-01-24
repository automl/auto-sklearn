import argparse
import os
import sys
import numpy as np

from sklearn.utils.multiclass import type_of_target
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy

import openml
#openml.config.cache_directory = os.path.join(os.path.expanduser("~"), 'openml') # Home directory. change this later accordingly
from remove_dataset_from_metadata import remove_dataset

def load_task(task_id):
    """Function used in score_vanilla and score_metalearning
    for loading data."""
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

def main(working_directory, time_limit, per_run_time_limit, task_id, seed):
    # Set to local dataset cache
    openml.config.cache_directory = os.path.join(working_directory, "cache")

    # Load data and other info.
    X_train, y_train, X_test, y_test, cat = load_task(task_id)
    # Check if data is dense or sparse.
    if isinstance(X_train, np.ndarray):
        sparse_or_dense = "dense"
    else:
        sparse_or_dense = "sparse"

    # path to the metadata directory. Is there ar better way to get this?
    metadata_directory = os.path.abspath(os.path.dirname(__file__))
    metadata_directory = os.path.join(metadata_directory, "../../autosklearn/metalearning/files/")
    #metadata_directory = os.path.dirname(autosklearn.metalearning.files.__file__)

    # Create new metadata directory not containing task_id.
    new_metadata_directory = os.path.abspath(os.path.join(working_directory, "metadata_%i" % task_id))

    try:
        os.makedirs(new_metadata_directory)
        remove_dataset(metadata_directory, new_metadata_directory, task_id)
    except:
        pass # pass because new metadata is created for this task.

    # We need to get task type, metric, is_sparse_or_dense information to
    # construct the path to the specific metadata directory. For details see
    # get_metalearning_suggestion() in smbo.py.
    TASK_TYPES_TO_STRING = {  # Mimic the same dict in autosklearn.constants
        'binary': 'binary.classification',
        'multiclass': 'multiclass.classification',
    }
    task_type = type_of_target(y_train)
    metadata_for_this_task = os.path.abspath(
        os.path.join(working_directory,
                     "metadata_%i/balanced_accuracy_%s_%s" % (task_id,
                                                              TASK_TYPES_TO_STRING[task_type],
                                                              sparse_or_dense)))
    # how to check if data is sparse before running?

    configuration_output_dir = os.path.join(working_directory, str(seed))
    tmp_dir = os.path.join(configuration_output_dir, str(task_id))
    try:
        if not os.path.exists(configuration_output_dir):
            os.makedirs(configuration_output_dir)
    except Exception as _:
        print("Direcotry {0} aleardy created.".format(configuration_output_dir))

    automl_arguments = {
        'time_left_for_this_task': time_limit,
        'per_run_time_limit': per_run_time_limit,
        'initial_configurations_via_metalearning': 25,
        'ensemble_size': 0,
        'seed': seed,
        'ml_memory_limit': 3072,
        'resampling_strategy': 'holdout',
        'resampling_strategy_arguments': {'train_size': 0.67},
        'tmp_folder': tmp_dir,
        'delete_tmp_folder_after_terminate': False,
        'disable_evaluator_output': False,
    }

    automl = AutoSklearnClassifier(**automl_arguments)
    # automl._automl._metadata_directory does not work cause automl._automl is not
    # created until fit is called. Therefore, we need to manually create
    # automl._automl and specify metadata_directory there.
    automl._automl = automl.build_automl()
    automl._automl._metadata_directory = metadata_for_this_task

    # Fit.
    automl._automl.fit(X_train,
                    y_train,
                    dataset_name=str(task_id),
                    X_test=X_test,
                    y_test=y_test,
                    metric=balanced_accuracy,
                    )

    with open(os.path.join(tmp_dir, "score_metalearning.csv"), 'w') as fh:
        T = 0
        fh.write("Time,Train Performance,Test Performance\n")
        # Add start time:0, Train Performance:1, Test Performance: 1
        fh.write("{0},{1},{2}\n".format(T, 1, 1))
        for t, dummy, s in zip(automl.cv_results_['mean_fit_time'],
                               [1 for i in range(len(automl.cv_results_['mean_fit_time']))],
                               1 - automl.cv_results_["mean_test_score"]):  # We compute rank based on error.
            T += t
            fh.write("{0},{1},{2}\n".format(T, dummy, s))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-directory', type=str, required=True)
    parser.add_argument('--time-limit', type=int, required=True)
    parser.add_argument('--per-run-time-limit', type=int, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('-s', '--seed', type=int, required=True)

    args = parser.parse_args()
    working_directory = args.working_directory
    time_limit = args.time_limit
    per_run_time_limit = args.per_run_time_limit
    task_id = args.task_id
    seed = args.seed

    main(working_directory, time_limit, per_run_time_limit, task_id, seed)
