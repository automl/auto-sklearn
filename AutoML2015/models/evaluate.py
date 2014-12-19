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
    X_train, X_optimization, Y_train, Y_optimization = \
        split_data(Datamanager.data['X_train'], Datamanager.data['Y_train'])

    model = AutoSklearnClassifier(configuration, 1)
    model.fit(X_train, Y_train)
    metric = Datamanager.info['metric']
    task_type = Datamanager.info['task']
    Y_pred = model.scores(X_optimization)

    if task_type == "multiclass.classification":
        Y_optimization_binary = np.zeros((Y_pred.shape))
        for i in range(Y_optimization_binary.shape[0]):
            label = Y_optimization[i]
            Y_optimization_binary[i, label] = 1
        Y_optimization = Y_optimization_binary

    if task_type == "multilabel.classification":
        Y_pred = np.hstack([Y_pred[i][:, 1].reshape((-1, 1))
                             for i in range(len(Y_pred))])

    scoring_func = getattr(libscores, metric)
    csolution, cprediction = libscores.normalize_array(Y_optimization, Y_pred)
    score = scoring_func(csolution, cprediction, task=task_type)

    err = 1 - score

    if with_predictions:
        return err, Y_pred
    else:
        return err
