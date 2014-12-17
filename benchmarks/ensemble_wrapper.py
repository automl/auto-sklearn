'''
Created on Dec 16, 2014

@author: Aaron Klein
'''

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from AutoSklearn.autosklearn import AutoSklearnClassifier

from HPOlibConfigSpace import configuration_space

from AutoML2015.util.split_data import split_data
from AutoML2015.data.data_manager import DataManager
from AutoML2015.scores import libscores

import ast
import time


def main(params, **kwargs):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass
    print params
    basename = "digits"
    input_dir = "/home/kleinaa/devel/git/automl2015/"
    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)
    print configuration

    D = DataManager(basename, input_dir, verbose=True)
    model = AutoSklearnClassifier(configuration)

    X_train, X_valid, Y_train, Y_valid = split_data(D.data['X_train'], D.data['Y_train'])

    model.fit(X_train, Y_train)
    Y_pred = model.scores(X_valid)

    metric = D.info['metric']
    task_type = D.info['task']

    scoring_func = getattr(libscores, metric)
    score = scoring_func(Y_valid, Y_pred, task=task_type)

    return 1 - score


if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
