'''
Created on Dec 17, 2014

@author: Aaron Klein
'''


try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli
from HPOlib.wrapping_util import get_time_string

try:
    import cPickle as pickle
except:
    import pickle

from AutoSklearn.autosklearn import AutoSklearnClassifier

from HPOlibConfigSpace import configuration_space

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import evaluate

import os
import time


def main(params, args):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    basename = args['dataset']
    input_dir = args['data_dir']
    output_dir = os.getcwd()

    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    D = DataManager(basename, input_dir, verbose=True)

    err, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
        evaluate(D, configuration, with_predictions=True)

    pred_dump_name_template = os.path.join(output_dir, "predictions_%s",
        basename + '_predictions_%s_' + get_time_string() + '.npy')

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

    return err

if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()

    result = main(params, args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
