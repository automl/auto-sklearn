import lockfile

import os
import time
import sys

try:
    import cPickle as pickle
except:
    import pickle



from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.autosklearn_regression import AutoSklearnRegressor

from HPOlibConfigSpace import configuration_space

from data.data_manager import DataManager

#from models.evaluate import evaluate

import models.evaluate
import signal



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
        #It is not yet sure, whether the file already exists
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
        print "Loaded data"
    return D





# signal handler seem to work only if they are globally defined
# to give it access to the evaluator class, the evaluator name has to 
# be a global name. It's not the cleanest solution, but works for now.
evaluator = None
def signal_handler(signum, frame):
    print "Aborting Training!"
    global evaluator
    evaluator.finish_up()
    exit(0)


signal.signal(15, signal_handler)


def main(basename, input_dir, params, time_limit=sys.maxint):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    output_dir = os.getcwd()
    D = store_and_or_load_data(data_dir=input_dir, dataset=basename,
                               outputdir=output_dir)

    if D.info["task"].lower() == "regression":
        cs = AutoSklearnRegressor.get_hyperparameter_search_space()
    else:
        cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    global evaluator
    evaluator = models.evaluate.evaluator(Datamanager=D, configuration=configuration, with_predictions=True, all_scoring_functions=True)
    evaluator.output_dir = output_dir
    evaluator.basename = basename
    evaluator.D = D
    evaluator.fit()

    evaluator.finish_up()

if __name__ == "__main__":
    outer_starttime = time.time()
    # Change a SMAC call into an HPOlib call, not yet needed!
    #if not "--params" in sys.argv:
    #    # Call from SMAC
    #    # Replace the SMAC seed by --params
    #    for i in range(len(sys.argv)):
    #        if sys.argv[i] == "2147483647" and sys.argv[i+1] == "-1":
    #            sys.argv[i+1] = "--params"

    limit = None
    for idx, arg in enumerate(sys.argv):
        if arg == "--limit":
            limit = int(float(sys.argv[idx+1]))
            del sys.argv[idx:idx+2]
            break

    instance_name = sys.argv[1]
    instance_specific_information = sys.argv[2]  # = 0
    cutoff_time = float(sys.argv[3])  # = inf
    cutoff_length = int(float(sys.argv[4]))  # = 2147483647
    seed = int(float(sys.argv[5]))

    if limit is None:
        limit = cutoff_time

    params = dict()
    for i in range(6, len(sys.argv), 2):
        p_name = str(sys.argv[i])
        if p_name[0].startswith("-"):
            p_name = p_name[1:]
        params[p_name] = sys.argv[i+1].strip()
    
    dataset = os.path.basename(instance_name);
    data_dir = os.path.dirname(instance_name);
    main(basename=dataset, input_dir=data_dir, params=params, time_limit=limit)

    sys.exit(0)
