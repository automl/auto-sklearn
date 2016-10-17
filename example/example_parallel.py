# -*- encoding: utf-8 -*-
import multiprocessing
import shutil

import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *

tmp_folder = '/tmp/autosklearn_parallel_example_tmp'
output_folder = '/tmp/autosklearn_parallel_example_out'


for dir in [tmp_folder, output_folder]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        pass


def spawn_classifier(seed, dataset_name):
    """Spawn a subprocess.

    auto-sklearn does not take care of spawning worker processes. This
    function, which is called several times in the main block is a new
    process which runs one instance of auto-sklearn.
    """

    # Use the initial configurations from meta-learning only in one out of
    # the four processes spawned. This prevents auto-sklearn from evaluating
    # the same configurations in four processes.
    if seed == 0:
        initial_configurations_via_metalearning = 25
    else:
        initial_configurations_via_metalearning = 0

    # Arguments which are different to other runs of auto-sklearn:
    # 1. all classifiers write to the same output directory
    # 2. shared_mode is set to True, this enables sharing of data between
    # models.
    # 3. all instances of the AutoSklearnClassifier must have a different seed!
    automl = AutoSklearnClassifier(
        time_left_for_this_task=120, # sec., how long should this seed fit
        # process run
        per_run_time_limit=60, # sec., each model may only take this long before it's killed
        ml_memory_limit=1024, # MB, memory limit imposed on each call to a ML  algorithm
        shared_mode=True, # tmp folder will be shared between seeds
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        delete_tmp_folder_after_terminate=False,
        ensemble_size=0, # ensembles will be built when all optimization runs are finished
        initial_configurations_via_metalearning=initial_configurations_via_metalearning,
        seed=seed)
    automl.fit(X_train, y_train, dataset_name=dataset_name)

if __name__ == '__main__':
    
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, random_state=1)

    processes = []
    for i in range(4): # set this at roughly half of your cores
        p = multiprocessing.Process(target=spawn_classifier, args=(i, 'digits'))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

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

    # Both the ensemble_size and ensemble_nbest parameters can be changed now if
    # necessary
    automl.fit_ensemble(y_train,
                        task=MULTICLASS_CLASSIFICATION,
                        metric=ACC_METRIC,
                        precision='32',
                        dataset_name='digits',
                        ensemble_size=20,
                        ensemble_nbest=50)

    predictions = automl.predict(X_test)
    print(automl.show_models())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
