import os
try:
    import cPickle as pickle
except:
    import pickle
import signal
import time

import lockfile

from HPOlibConfigSpace import configuration_space

from autosklearn.data.data_manager import DataManager
from autosklearn.models.evaluator import get_new_run_num
from autosklearn.models.holdout_evaluator import HoldoutEvaluator
from autosklearn.models.cv_evaluator import CVEvaluator
from autosklearn.models.test_evaluator import TestEvaluator
from autosklearn.models.nested_cv_evaluator import NestedCVEvaluator
from autosklearn.models.paramsklearn import get_configuration_space


def store_and_or_load_data(outputdir, dataset, data_dir):
    save_path = os.path.join(outputdir, dataset + "_Manager.pkl")
    if not os.path.exists(save_path):
        lock = lockfile.LockFile(save_path)
        while not lock.i_am_locking():
            try:
                lock.acquire(timeout=60)  # wait up to 60 seconds
            except lockfile.LockTimeout:
                lock.break_lock()
                lock.acquire()
        print "I locked", lock.path
        # It is not yet sure, whether the file already exists
        try:
            if not os.path.exists(save_path):
                D = DataManager(dataset, data_dir, verbose=True,
                                encode_labels=True)
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


# signal handler seem to work only if they are globally defined
# to give it access to the evaluator class, the evaluator name has to
# be a global name. It's not the cleanest solution, but works for now.
evaluator = None

def signal_handler(signum, frame):
    print "Aborting Training!"
    global evaluator
    evaluator.finish_up()
    exit(0)

def empty_signal_handler(signum, frame):
    print "Received Signal %s, but alread finishing up!" % str(signum)

signal.signal(15, signal_handler)


def main(dataset, data_dir, mode, seed, params, mode_args=None):
    """This command line interface has three different operation modes:

    * CV: useful for the Tweakathon
    * 1/3 test split: useful to evaluate a configuration
    * cv on 2/3 train split: useful to optimize hyperparameters in a training
      mode before testing a configuration on the 1/3 test split.

    It must by no means be used for the Auto part of the competition!
    """
    if mode != "test":
        num_run = get_new_run_num()

    for key in params:
        try:
            params[key] = int(params[key])
        except:
            try:
                params[key] = float(params[key])
            except:
                pass

    if seed is not None:
        seed = int(float(seed))
    else:
        seed = 1

    output_dir = os.getcwd()

    D = store_and_or_load_data(data_dir=data_dir,
                               dataset=dataset,
                               outputdir=output_dir)

    cs = get_configuration_space(D.info)
    configuration = configuration_space.Configuration(cs, params)
    metric = D.info['metric']

    global evaluator
    # Train/test split
    if mode == 'holdout':
        evaluator = HoldoutEvaluator(D, configuration,
                                     with_predictions=True,
                                     all_scoring_functions=True,
                                     output_y_test=True,
                                     seed=seed, num_run=num_run)
        evaluator.fit()
        signal.signal(15, empty_signal_handler)
        evaluator.finish_up()

    elif mode == 'test':
        evaluator = TestEvaluator(D, configuration,
                                  all_scoring_functions=True,
                                  seed=seed)
        evaluator.fit()
        scores = evaluator.predict()
        duration = time.time() - evaluator.starttime

        score = scores[metric]
        additional_run_info = ";".join(["%s: %s" % (m_, value)
                                        for m_, value in scores.items()])
        additional_run_info += ";" + "duration: " + str(duration)

        print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % (
            "SAT", abs(duration), score, evaluator.seed, additional_run_info)

    # CV on the whole training set
    elif mode == 'cv':
        evaluator = CVEvaluator(D, configuration, with_predictions=True,
                                all_scoring_functions=True, output_y_test=True,
                                cv_folds=mode_args['folds'], seed=seed,
                                num_run=num_run)
        evaluator.fit()
        signal.signal(15, empty_signal_handler)
        evaluator.finish_up()

    elif mode == 'partial_cv':
        evaluator = CVEvaluator(D, configuration, all_scoring_functions=True,
                                cv_folds=mode_args['folds'], seed=seed,
                                num_run=num_run)
        evaluator.partial_fit(mode_args['fold'])
        scores = evaluator.predict()
        duration = time.time() - evaluator.starttime

        score = scores[metric]
        additional_run_info = ";".join(["%s: %s" % (m_, value)
                                        for m_, value in scores.items()])
        additional_run_info += ";" + "duration: " + str(duration)

        print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % (
            "SAT", abs(duration), score, evaluator.seed, additional_run_info)

    elif mode == 'nested-cv':
        evaluator = NestedCVEvaluator(D, configuration, with_predictions=True,
                                      inner_cv_folds=mode_args['inner_folds'],
                                      outer_cv_folds=mode_args['outer_folds'],
                                      all_scoring_functions=True,
                                      output_y_test=True, seed=seed,
                                      num_run=num_run)
        evaluator.fit()
        signal.signal(15, empty_signal_handler)
        evaluator.finish_up()

    else:
        raise ValueError("Must choose a legal mode.")