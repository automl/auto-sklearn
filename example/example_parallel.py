import logging
import multiprocessing

import numpy as np

import sklearn.datasets
import sklearn.metrics

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *

tmp_folder = '/tmp/autoslearn_example_tmp'
output_folder = '/tmp/autosklearn_example_out'


def spawn_classifier(seed, dataset_name):
    """Spawn a subprocess.

    auto-sklearn does not take care of spawning worker processes. This
    function, which is called several times in the main block is a new
    process which runs one instance of auto-sklearn.
    """
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train = X[:1000]
    y_train = y[:1000]

    # Arguments which are different to other runs of auto-sklearn:
    # 1. all classifiers write to the same output directory
    # 2. shared_mode is set to True, this enables sharing of data between
    # models.
    # 3. all instances of the AutoSklearnClassifier must have a different seed!
    automl = AutoSklearnClassifier(time_left_for_this_task=60,
                                   per_run_time_limit=60,
                                   ml_memory_limit=1024,
                                   shared_mode=True,
                                   tmp_folder=tmp_folder,
                                   output_folder=output_folder,
                                   delete_tmp_folder_after_terminate=False,
                                   ensemble_size=0,
                                   initial_configurations_via_metalearning=0,
                                   seed=seed)
    automl.fit(X_train, y_train, dataset_name=dataset_name)

if __name__ == '__main__':
    processes = []
    for i in range(2, 6):
        p = multiprocessing.Process(target=spawn_classifier, args=(i, 'digits'))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1000:]
    y_test = y[1000:]

    print('Starting to build an ensemble!')
    automl = AutoSklearnClassifier(time_left_for_this_task=15,
                                   per_run_time_limit=15,
                                   ml_memory_limit=1024,
                                   shared_mode=True,
                                   ensemble_size=50,
                                   ensemble_nbest=200,
                                   tmp_folder=tmp_folder,
                                   output_folder=output_folder,
                                   initial_configurations_via_metalearning=0,
                                   seed=1)

    # Both the ensemble_size and ensemble_nbest parameters can be changed later
    automl.fit_ensemble(task=MULTICLASS_CLASSIFICATION,
                        metric=ACC_METRIC,
                        precision='32',
                        dataset_name='digits',
                        ensemble_size=10,
                        ensemble_nbest=10)

    predictions = automl.predict(X_test)
    print(automl.show_models())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))