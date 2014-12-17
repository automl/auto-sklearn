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

import time
import numpy as np


def main(params, **kwargs):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass
    print params

    basename = "digits"
    input_dir = "/home/feurerm/projects/automl_competition_2015/datasets/000"
    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)
    print configuration

    D = DataManager(basename, input_dir, verbose=True)
    print D
    X_train, X_valid, Y_train, Y_valid = split_data(D.data['X_train'],
                                                    D.data['Y_train'])

    model = AutoSklearnClassifier(configuration, 1)
    model.fit(X_train, Y_train)
    metric = D.info['metric']
    task_type = D.info['task']
    Y_pred = model.scores(X_valid)

    if task_type == "multiclass.classification":
        Y_valid_binary = np.zeros((Y_pred.shape))
        for i in range(Y_valid_binary.shape[0]):
            label = Y_valid[i]
            Y_valid_binary[i,label] = 1
        Y_valid = Y_valid_binary

    scoring_func = getattr(libscores, metric)
    csolution, cprediction = libscores.normalize_array(Y_valid, Y_pred)
    score = scoring_func(csolution, cprediction, task=task_type)
    all_scores = libscores.compute_all_scores(Y_valid, Y_pred)
    print all_scores
    return 1 - score


if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
