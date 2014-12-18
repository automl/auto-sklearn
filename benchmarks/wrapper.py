'''
Created on Dec 17, 2014

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

    score = evaluate(D, configuration)

    return score

if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()

    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
