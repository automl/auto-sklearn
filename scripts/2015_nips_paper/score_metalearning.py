import argparse
import os
import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy

from remove_dataset_from_metadata import remove_dataset
sys.path.append('../')
from update_metadata_util import load_task  # noqa
#from openml.tasks import get_task

def main(working_directory, time_limit, per_run_time_limit, task_id, seed):
    # Create new metadata directory not containing task_id.
    metadata_directory = os.path.abspath(os.path.dirname(__file__))
    metadata_directory = os.path.join(metadata_directory, "../../autosklearn/metalearning/files/")
    new_metadata_directory = os.path.abspath(os.path.join(working_directory, "metadata_{}".format(task_id)))
    try:
        os.makedirs(new_metadata_directory)
        remove_dataset(metadata_directory, new_metadata_directory, task_id)
    except:
        pass # pass because new metadata is created for this task.

    configuration_output_dir = os.path.join(working_directory, str(seed))
    try:
        os.makedirs(configuration_output_dir)
    except Exception as _:
        print("Direcotry {0} aleardy created.".format(configuration_output_dir))

    tmp_dir = os.path.join(configuration_output_dir, str(task_id))

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

    X_train, y_train, X_test, y_test, cat = load_task(task_id)

    automl = AutoSklearnClassifier(**automl_arguments)
    print(automl._automl._metadata_directory)
    print("####")
    return

    automl.fit(X_train, y_train,
               dataset_name=str(task_id),
               X_test=X_test, y_test=y_test,
               metric=balanced_accuracy)

    with open(os.path.join(tmp_dir, "score_vanilla.csv"), 'w') as fh:
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
