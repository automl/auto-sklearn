'''
Created on Dec 18, 2014

@author: Aaron Klein
'''

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from AutoSklearn.autosklearn import AutoSklearnClassifier

from HPOlibConfigSpace import configuration_space

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import evaluate

import time


def main(params, **kwargs):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    basename = args['dataset']
    input_dir = args['data_dir']

    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    D = DataManager(basename, input_dir, verbose=True)

    score, predictions = evaluate(D, configuration, with_predictions=True)

    return score, predictions

if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()
    result, predictions = main(params, **args)
    duration = time.time() - starttime

    run_info = str(result) + ";"
    for p in predictions:
        for i in p:
            run_info += str(i) + ";"
    run_info = run_info[:-1]

    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, run_info)
