'''
Created on Jan 30, 2015

@author: Aaron Klein
'''


import lockfile
import os
import time
import numpy as np
from functools import partial

try:
    import cPickle as pickle
except:
    import pickle
from AutoSklearn.classification import AutoSklearnClassifier
from AutoSklearn.regression import AutoSklearnRegressor

from HPOlibConfigSpace import configuration_space

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import Evaluator
from AutoML2015.util.split_data import get_CV_fold


def store_and_or_load_data(outputdir, dataset, data_dir):
    save_path = os.path.join(outputdir, dataset + "_Manager.pkl")
    if not os.path.exists(save_path):
        lock = lockfile.LockFile(save_path)
        while not lock.i_am_locking():
            try:
                lock.acquire(timeout=60)    # wait up to 60 seconds
            except lockfile.LockTimeout:
                lock.break_lock()
                lock.acquire()
        print "I locked", lock.path
        # It is not yet sure, whether the file already exists
        try:
            if not os.path.exists(save_path):
                D = DataManager(dataset, data_dir, verbose=True)
                fh = open(save_path, 'w')
                pickle.dump(D, fh, -1)
                fh.close()
            else:
                D = pickle.load(open(save_path, 'r'))
        except:
            raise
        finally:
            lock.release()
    else:
        D = pickle.load(open(save_path, 'r'))
    return D


def get_new_run_num():
    counter_file = os.path.join(os.getcwd(), "num_run")
    lock = lockfile.LockFile(counter_file)
    with lock:
        if not os.path.exists(counter_file):
            with open(counter_file, "w") as fh:
                fh.write("0")
            num = 0
        else:
            with open(counter_file, "r") as fh:
                num = int(fh.read())
            num += 1
            with open(counter_file, "w") as fh:
                fh.write(str(num).zfill(4))

    return num


def main(args, params):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    #FIXME: If we make the CV internally such that SMAC does not know it, we do not need this variable right?
    fold = int(args['fold'])
    folds = int(args['folds'])
    folds = 4

    basename = args['dataset']
    input_dir = args['data_dir']
    output_dir = os.getcwd()

    D = store_and_or_load_data(data_dir=input_dir, dataset=basename,
                               outputdir=output_dir)

    if D.info['task'].lower() == 'regression':
        cs = AutoSklearnRegressor.get_hyperparameter_search_space()
    else:
        cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    starttime = time.time()

    def splitting_function(X, Y, fold, folds):
        train_indices, test_indices = \
        get_CV_fold(X, Y, fold=fold, folds=folds)
        return X[train_indices], X[test_indices], Y[train_indices], \
            Y[test_indices]

    errors = []
    Y_optimization_pred = []
    Y_valid_pred = []
    Y_test_pred = []
    for k in range(folds):
        splitting_function = partial(splitting_function, fold=k, folds=folds)

        evaluator = Evaluator(D, configuration, with_predictions=True,
                          all_scoring_functions=True, splitting_function=splitting_function)
        evaluator.fit()
        err, opt_pred, valid_pred, test_pred = \
            evaluator.predict()
        errors.append(err)
        Y_optimization_pred.append(opt_pred)
        Y_valid_pred.append(valid_pred)
        Y_test_pred.append(test_pred)

    duration = time.time() - starttime

    Y_optimization_pred = np.array(Y_optimization_pred)
    Y_optimization_pred = np.reshape(Y_optimization_pred, (Y_optimization_pred.shape[1] * folds, Y_optimization_pred.shape[2]))
    Y_valid_pred = np.array(Y_valid_pred)
    Y_valid_pred = np.reshape(Y_valid_pred, (Y_valid_pred.shape[1] * folds, Y_valid_pred.shape[2]))
    Y_test_pred = np.array(Y_test_pred)
    Y_test_pred = np.reshape(Y_test_pred, (Y_test_pred.shape[1] * folds, Y_test_pred.shape[2]))

    pred_dump_name_template = os.path.join(output_dir, "predictions_%s",
        basename + '_predictions_%s_' + str(get_new_run_num()) + '.npy')

    ensemble_output_dir = os.path.join(output_dir, "predictions_ensemble")
    if not os.path.exists(ensemble_output_dir):
        os.mkdir(ensemble_output_dir)
    with open(pred_dump_name_template % ("ensemble", "ensemble"), "w") as fh:
        pickle.dump(Y_optimization_pred, fh, -1)

    valid_output_dir = os.path.join(output_dir, "predictions_valid")
    if not os.path.exists(valid_output_dir):
        os.mkdir(valid_output_dir)
    with open(pred_dump_name_template % ("valid", "valid"), "w") as fh:
        pickle.dump(Y_valid_pred, fh, -1)

    test_output_dir = os.path.join(output_dir, "predictions_test")
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)
    with open(pred_dump_name_template % ("test", "test"), "w") as fh:
        pickle.dump(Y_test_pred, fh, -1)

    # Compute the average over all folds for each scoring function
    errs = {key: ((1 / float(folds)) * errors[0][key]) for key in errors[0].keys()}
    for i in range(1, folds):
        for key in errors[i].keys():
            errs[key] += errors[i][key] * (1 / float(folds))

    print errs
    err = errs[D.info['metric']]
    import sys
    sys.stdout.flush()
    additional_run_info = ";".join(["%s: %s" % (metric, value)
                                    for metric, value in errs.items()])
    additional_run_info += ";" + "duration: " + str(duration)
    additional_run_info += ";" + "prediction_files_template: " + \
        pred_dump_name_template
    return err, additional_run_info


if __name__ == "__main__":
    starttime = time.time()
    # Change a SMAC call into an HPOlib call, not yet needed!
    #if not "--params" in sys.argv:
    #    # Call from SMAC
    #    # Replace the SMAC seed by --params
    #    for i in range(len(sys.argv)):
    #        if sys.argv[i] == "2147483647" and sys.argv[i+1] == "-1":
    #            sys.argv[i+1] = "--params"

    args, params = parse_cli()

    result, additional_run_info = main(args, params)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, additional_run_info)
