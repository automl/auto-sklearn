try:
    import cPickle as pickle
except Exception:
    import pickle
# -*- encoding: utf-8 -*-

import os
import signal
import sys
import time

import autosklearn.models.holdout_evaluator
import lockfile
from autosklearn.data.data_manager import DataManager
from autosklearn.models.paramsklearn import get_class
from HPOlibConfigSpace import configuration_space


def store_and_or_load_data(outputdir, dataset, data_dir):
    save_path = os.path.join(outputdir, dataset + '_Manager.pkl')
    if not os.path.exists(save_path):
        lock = lockfile.LockFile(save_path)
        while not lock.i_am_locking():
            try:
                lock.acquire(timeout=60)  # wait up to 60 seconds
            except lockfile.LockTimeout:
                lock.break_lock()
                lock.acquire()
        print('I locked', lock.path)
        # It is not yet sure, whether the file already exists
        try:
            if not os.path.exists(save_path):
                D = DataManager(dataset, data_dir, verbose=True)
                fh = open(save_path, 'w')
                pickle.dump(D, fh, -1)
                fh.close()
            else:
                D = pickle.load(open(save_path, 'r'))
        except Exception:
            raise
        finally:
            lock.release()
    else:
        D = pickle.load(open(save_path, 'r'))
        print('Loaded data')
    return D

# signal handler seem to work only if they are globally defined
# to give it access to the evaluator class, the evaluator name has to
# be a global name. It's not the cleanest solution, but works for now.
evaluator = None


def signal_handler(signum, frame):
    print('Aborting Training!')
    global evaluator
    evaluator.finish_up()
    exit(0)


signal.signal(15, signal_handler)


def main(basename, input_dir, params):

    output_dir = os.getcwd()
    D = store_and_or_load_data(data_dir=input_dir,
                               dataset=basename,
                               outputdir=output_dir)

    cs = get_class(D.info).get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    global evaluator
    evaluator = autosklearn.models.holdout_evaluator.HoldoutEvaluator(
        datamanager=D, configuration=configuration, with_predictions=True,
        all_scoring_functions=True, output_dir=output_dir)
    evaluator.fit()
    evaluator.finish_up()


if __name__ == '__main__':
    outer_starttime = time.time()

    instance_name = sys.argv[1]
    instance_specific_information = sys.argv[2]  # = 0
    cutoff_time = float(sys.argv[3])  # = inf
    cutoff_length = int(float(sys.argv[4]))  # = 2147483647
    seed = int(float(sys.argv[5]))

    params = dict()
    for i in range(6, len(sys.argv), 2):
        p_name = str(sys.argv[i])
        if p_name[0].startswith('-'):
            p_name = p_name[1:]
        params[p_name] = sys.argv[i + 1].strip()

    for key in params:
        try:
            params[key] = float(params[key])
        except Exception:
            pass

    dataset = os.path.basename(instance_name)
    data_dir = os.path.dirname(instance_name)
    main(basename=dataset, input_dir=data_dir, params=params)

    sys.exit(0)
