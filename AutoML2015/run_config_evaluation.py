import lockfile

import os
import time
import sys

try:
    import cPickle as pickle
except:
    import pickle
from AutoSklearn.autosklearn import AutoSklearnClassifier

from HPOlibConfigSpace import configuration_space

from data.data_manager import DataManager
from models.evaluate import evaluate


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
            fh.write(str(num))
        return num


def main(basename, input_dir, params, time_limit=sys.maxint):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    output_dir = os.getcwd()

    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    D = store_and_or_load_data(data_dir=input_dir, dataset=basename,
                               outputdir=output_dir)

    starttime = time.time()
    errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = evaluate(
        Datamanager=D, configuration=configuration, with_predictions=True,
        all_scoring_functions=True)
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
    outer_starttime = time.time()
    # Change a SMAC call into an HPOlib call, not yet needed!
    #if not "--params" in sys.argv:
    #    # Call from SMAC
    #    # Replace the SMAC seed by --params
    #    for i in range(len(sys.argv)):
    #        if sys.argv[i] == "2147483647" and sys.argv[i+1] == "-1":
    #            sys.argv[i+1] = "--params"
    instance_name = sys.argv[1]
    instance_specific_information = sys.argv[2] # = 0
    cutoff_time = float(sys.argv[3]) # = inf
    cutoff_length = int(float(sys.argv[4])) # = 2147483647
    seed = int(float(sys.argv[5]))

    params = dict()
    for i in range(6, len(sys.argv), 2):
        p_name = str(sys.argv[i])
        if p_name[0].startswith("-"):
            p_name = p_name[1:]
        params[p_name] = sys.argv[i+1].strip()

    dataset = os.path.basename(instance_name)
    data_dir = os.path.dirname(instance_name)

    result, additional_run_info = main(basename=dataset, input_dir=data_dir,
                                       params=params, time_limit=cutoff_time)
    outer_duration = time.time() - outer_starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(outer_duration), result, seed, additional_run_info)
    sys.exit(0)
