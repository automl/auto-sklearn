'''
Created on Dec 18, 2014

@author: Aaron Klein
'''

import cPickle
import numpy as np

from AutoSklearn.autosklearn import AutoSklearnClassifier

from AutoML2015.util.split_data import split_data
from AutoML2015.scores import libscores


def evaluate(Datamanager, configuration, with_predictions=False):
    X_train, X_valid, Y_train, Y_valid = split_data(Datamanager.data['X_train'],
                                                    Datamanager.data['Y_train'])

    model = AutoSklearnClassifier(configuration, 1)
    model.fit(X_train, Y_train)
    metric = Datamanager.info['metric']
    task_type = Datamanager.info['task']
    Y_pred = model.scores(X_valid)

    if task_type == "multiclass.classification":
        Y_valid_binary = np.zeros((Y_pred.shape))
        for i in range(Y_valid_binary.shape[0]):
            label = Y_valid[i]
            Y_valid_binary[i, label] = 1
        Y_valid = Y_valid_binary

    scoring_func = getattr(libscores, metric)
    csolution, cprediction = libscores.normalize_array(Y_valid, Y_pred)
    score = scoring_func(csolution, cprediction, task=task_type)

    err = 1 - score

    if with_predictions:
        return err, Y_pred
    else:
        return err
