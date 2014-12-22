'''
Created on Dec 18, 2014

@author: Aaron Klein
'''

import cPickle
import numpy as np

from AutoSklearn.autosklearn import AutoSklearnClassifier

from AutoML2015.data.data_converter import convert_to_bin
from AutoML2015.scores import libscores
from AutoML2015.util.split_data import split_data


def predict_proba(X, model, task_type):
    Y_pred = model.predict_proba(X)

    if task_type == "multilabel.classification":
        Y_pred = np.hstack(
            [Y_pred[i][:, 1].reshape((-1, 1))
             for i in range(len(Y_pred))])

    elif task_type == "binary.classification":
        if len(Y_pred.shape) == 1:
            Y_pred = convert_to_bin(Y_pred, 2)

    return Y_pred


def evaluate(Datamanager, configuration, with_predictions=False):
    X_train, X_optimization, Y_train, Y_optimization = \
        split_data(Datamanager.data['X_train'], Datamanager.data['Y_train'])
    X_valid = Datamanager.data['X_valid']
    X_test = Datamanager.data['X_test']

    model = AutoSklearnClassifier(configuration, 1)
    model.fit(X_train, Y_train)
    metric = Datamanager.info['metric']
    task_type = Datamanager.info['task']

    Y_optimization_pred = \
        predict_proba(X_optimization, model, task_type)

    if task_type == "multiclass.classification":
        Y_optimization_binary = np.zeros((Y_optimization_pred.shape))
        for i in range(Y_optimization_binary.shape[0]):
            label = Y_optimization[i]
            Y_optimization_binary[i, label] = 1
        Y_optimization = Y_optimization_binary

    elif task_type == "binary.classification":
        if len(Y_optimization.shape) == 1:
            Y_optimization = convert_to_bin(Y_optimization, 2)

    scoring_func = getattr(libscores, metric)
    csolution, cprediction = libscores.normalize_array(Y_optimization,
                                                       Y_optimization_pred)
    score = scoring_func(csolution, cprediction, task=task_type)

    Y_valid_pred = predict_proba(X_valid, model, task_type)
    Y_test_pred = predict_proba(X_test, model, task_type)

    err = 1 - score

    if with_predictions:
        return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
    else:
        return err
