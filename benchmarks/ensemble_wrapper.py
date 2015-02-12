'''
Created on Dec 18, 2014

@author: Aaron Klein
'''

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from AutoSklearn.classification import AutoSklearnClassifier
from AutoSklearn.regression import AutoSklearnRegressor


from HPOlibConfigSpace import configuration_space

from AutoML2015.util.get_dataset_info import getInfoFromFile
from AutoML2015.util.split_data import split_data
from AutoML2015.data.data_manager import DataManager
from AutoML2015.models import evaluate

import time
import os
import numpy as np


def ensemble_prediction(all_predictions, weights):
    pred = np.zeros([all_predictions.shape[1], all_predictions.shape[2]])
    for i, w in enumerate(weights):
        pred += all_predictions[i] * w

    return pred


def main(params, **kwargs):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    basename = args['dataset']
    input_dir = args['data_dir']
    D = DataManager(basename, input_dir, verbose=True)
    info = getInfoFromFile(input_dir, basename)

    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)
    model = AutoSklearnClassifier(configuration, random_state=np.random.randint(1000))
    X_train, X_valid, Y_train, Y_valid = split_data(D.data['X_train'], D.data['Y_train'])
    model.fit(X_train, Y_train)
    new_predictions = evaluate.predict_proba(X_valid, model, info['task'])
    if os.path.isfile("predictions.npy"):
        previous_predictions = np.load("predictions.npy")
        all_predictions = np.concatenate((previous_predictions, np.array([new_predictions])), axis=0)
    else:
        all_predictions = np.array([new_predictions])

    if os.path.isfile("weights.txt"):
        weights = np.loadtxt("weights.txt", delimiter=",")
    else:
        weights = np.ones([1])

    weighted_predictions = ensemble_prediction(all_predictions, weights)
    score = evaluate.calculate_score(Y_valid, weighted_predictions,
                                     info['task'], info['metric'])
    np.save("predictions.npy", all_predictions)
    return (1 - score), new_predictions


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
