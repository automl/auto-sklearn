'''
Created on Dec 17, 2014

@author: Aaron Klein
'''

import lockfile

import os
import sys
import time

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
from HPOlib.wrapping_util import get_time_string

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import Evaluator


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
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as fh:
            fh.write("0")
        return 0
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

    evaluator = Evaluator(D, configuration, with_predictions=True,
                          all_scoring_functions=True)
    evaluator.fit()
    errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
        evaluator.predict()
    duration = time.time() - starttime

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

    err = errs[D.info['metric']]
    print errs
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
