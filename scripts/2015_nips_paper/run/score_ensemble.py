from argparse import ArgumentParser
import csv
import glob
import os
import time

import numpy as np
import sklearn.externals.joblib as joblib

from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import balanced_accuracy

from autosklearn.util.backend import create


def _load_file(f):
    split = f.split('_')
    as_seed = int(split[-2])
    ta_seed = int(split[-1].split('.')[0])
    np_array = np.load(f)
    return np_array, (as_seed, ta_seed), os.path.getmtime(f)


def read_files(directory, seed=None, n_jobs=1):

    seed_pattern = '*' if seed is None else str(seed)
    glob_pattern = os.path.join(directory, "predictions_*_%s_*.npy" %
                                seed_pattern)
    files = sorted(glob.glob(glob_pattern))
    files = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(_load_file)(f=f) for f in files)

    return files


def main(input_directories, output_file, task_id, seed, ensemble_size, n_jobs=1):
    seed = None if seed is None or seed < 0 else int(seed)

    if isinstance(input_directories, str):
        # add seed and task id directories
        input_directories += '/%i/%i' % (seed, task_id)
        input_directories = [input_directories]

    else:
        new_directories = []
        for dir in input_directories:
            dir += '/%i/%i' % (seed, task_id)
            new_directories.append(dir)
        input_directories = new_directories

    validation_files = []
    test_files = []
    starttime = np.inf

    # Get the prediction files.
    for input_directory in input_directories:
        print('Loading files from input directory:', input_directory)
        validation_files_ = read_files(
            os.path.join(input_directory,
                         '.auto-sklearn/predictions_ensemble'),
            n_jobs=n_jobs)
        validation_files.extend(validation_files_)
        test_files_ = read_files(
            os.path.join(input_directory,
                         '.auto-sklearn/predictions_test'),
            n_jobs=n_jobs)
        test_files.extend(test_files_)

        assert len(validation_files_) > 0
        assert len(validation_files_) == len(test_files_)

        print('Loaded %d files!' % len(validation_files_))

        # if not specified, we get all files.
        seed_pattern = '*' if seed is None else str(seed)
        glob_pattern = os.path.join(input_directory,
                                    ".auto-sklearn",
                                    "start_time_%s" % seed_pattern)
        start_time_files = glob.glob(glob_pattern)

        # find the earliest startime.
        for start_time_file in start_time_files:
            with open(start_time_file, "r") as fh:
                starttime_candidate = float(fh.read())
                if starttime_candidate < starttime:
                    starttime = starttime_candidate

        del validation_files_
        del test_files_

    validation_files.sort(key=lambda t: t[-1])

    keys_to_test_files = {test_file[1]: test_file
                          for test_file in test_files}
    # Resort such that both files have the same order
    test_files = [keys_to_test_files[validation_file[1]]
                  for validation_file in validation_files]

    assert [validation_file[1] for validation_file in validation_files] == [
        test_file[1] for test_file in test_files]

    losses = []
    top_models_at_step = dict()

    backend = create(input_directory, input_directory + "_output",
                     delete_tmp_folder_after_terminate=False,
                     delete_output_folder_after_terminate=True,
                     shared_mode=True)
    valid_labels = backend.load_targets_ensemble()
    score = balanced_accuracy

    # Compute losses and remember best model at each time step.
    for i in range(len(validation_files)):
        loss = 1 - score(valid_labels, validation_files[i][0])
        losses.append(loss)
        sorted_losses = np.argsort(losses)[:200]
        top_models_at_step[i] = sorted_losses

    models_to_remove = set(list(range(len(validation_files))))
    for top_models_at_i in top_models_at_step:
        for top_model in top_models_at_step[top_models_at_i]:
            if top_model in models_to_remove:
                models_to_remove.remove(top_model)

    print("Removing the following %d models from the library: %s"
          % (len(models_to_remove), models_to_remove))
    for model_id in models_to_remove:
        validation_files[model_id] = None
        test_files[model_id] = None

    print('Starting ensemble building!')
    output = joblib.Parallel(n_jobs=n_jobs, verbose=20)(
        joblib.delayed(
            evaluate)(input_directory=input_directories[0],
                      validation_files=[validation_files[j] for
                                        j in range(len(validation_files))
                                        if j in top_models_at_step[i]],
                      test_files=[test_files[j] for
                                  j in range(len(test_files))
                                  if j in top_models_at_step[i]],
                      ensemble_size=ensemble_size)
        for i in range(len(test_files)))

    # Create output csv file
    file_path = os.path.abspath("%s/%s" % (input_directory, output_file))
    with open(file_path, "w") as csv_file:
        fieldnames = ['Time', 'Training (Empirical) Performance',
                      'Test Set Performance']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        # First time step
        csv_writer.writerow({'Time': 0,
                             'Training (Empirical) Performance': 1.0,
                             'Test Set Performance': 1.0})

        for i, o in enumerate(output):
            csv_writer.writerow({'Time': o['ensemble_time']
                                 + o['time_function_evaluation']
                                 - starttime,
                                 'Training (Empirical) Performance':
                                 o['ensemble_error'],
                                 'Test Set Performance':
                                 o['ensemble_test_error']})


def evaluate(input_directory, validation_files, test_files, ensemble_size=50):

    backend = create(input_directory, input_directory + "_output",
                     delete_tmp_folder_after_terminate=False,
                     delete_output_folder_after_terminate=False,
                     shared_mode=True)

    valid_labels = backend.load_targets_ensemble()
    D = backend.load_datamanager()
    test_labels = D.data["Y_test"]

    score = balanced_accuracy

    # Read the modification time of the predictions file and
    # compute the interval to the first prediction file.
    # This interval will be add to the time we needed to build the ensemble
    time_function_evaluation = validation_files[-1][-1]

    # Build the ensemble
    start = time.time()
    ensemble_selection = EnsembleSelection(ensemble_size=ensemble_size,
                                           task_type=D.info['task'],
                                           metric=score,
                                           random_state=np.random.RandomState())

    validation_predictions = np.array([v[0] for v in validation_files])
    test_predictions = np.array([t[0] for t in test_files])

    ensemble_selection.fit(validation_predictions, valid_labels,
                           identifiers=None)
    y_hat_ensemble = ensemble_selection.predict(np.array(
        validation_predictions))
    y_hat_test = ensemble_selection.predict(np.array(test_predictions))

    # Compute validation error
    ensemble_error = 1 - score(valid_labels, y_hat_ensemble)

    # Compute test error
    ensemble_test_error = 1 - score(test_labels, y_hat_test)

    ensemble_time = time.time() - start

    rval = {'ensemble_time': ensemble_time,
            'time_function_evaluation': time_function_evaluation,
            'ensemble_error': ensemble_error,
            'ensemble_test_error': ensemble_test_error}

    return rval


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-directory', type=str,
                        required=True, nargs='+')
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument("--output-file", type=str, default='score_ensemble.csv')
    parser.add_argument("--ensemble-size", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    input_directory = args.input_directory  # logdir/vanilla or logdir/metalearning
    output_file = args.output_file
    task_id = args.task_id
    seed = args.seed
    ensemble_size = args.ensemble_size
    n_jobs = args.n_jobs

    main(input_directory, output_file, task_id, seed, ensemble_size)
