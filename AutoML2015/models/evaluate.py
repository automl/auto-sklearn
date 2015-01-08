'''
Created on Dec 18, 2014

@author: Aaron Klein
'''

import numpy as np

from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.autosklearn_regression import AutoSklearnRegressor

from data.data_converter import convert_to_bin
from scores import libscores
from util.split_data import split_data


def predict_proba(X, model, task_type):
    Y_pred = model.predict_proba(X)

    if task_type == "multilabel.classification":
        Y_pred = np.hstack(
            [Y_pred[i][:, 1].reshape((-1, 1))
             for i in range(len(Y_pred))])

    elif task_type == "binary.classification":
        if len(Y_pred.shape) != 1:
            Y_pred = Y_pred[:,1].reshape(-1, 1)

    return Y_pred


def calculate_score(solution, prediction, task_type, metric,
                    all_scoring_functions=False):
    if task_type == "multiclass.classification":
        solution_binary = np.zeros((prediction.shape))
        for i in range(solution_binary.shape[0]):
            label = solution[i]
            solution_binary[i, label] = 1
        solution = solution_binary

    elif task_type == "binary.classification":
        if len(solution.shape) == 1:
            solution = solution.reshape((-1, 1))

    scoring_func = getattr(libscores, metric)

    csolution, cprediction = libscores.normalize_array(solution, prediction)
    if all_scoring_functions:
        score = dict()
        score["bac_metric"] = libscores.bac_metric(csolution, cprediction,
                                                   task=task_type)
        score["auc_metric"] = libscores.auc_metric(csolution, cprediction,
                                                   task=task_type)
        score["f1_metric"] = libscores.f1_metric(csolution, cprediction,
                                                 task=task_type)
        score["pac_metric"] = libscores.pac_metric(csolution, cprediction,
                                                   task=task_type)
        score["a_metric"] = libscores.a_metric(csolution, cprediction,
                                               task=task_type)
        score["r2_metric"] = libscores.r2_metric(csolution, cprediction,
                                                 task=task_type)
    else:
        score = scoring_func(csolution, cprediction, task=task_type)
    return score


def evaluate(Datamanager, configuration, with_predictions=False,
        all_scoring_functions=False, splitting_function=split_data, seed=1):
    X_train, X_optimization, Y_train, Y_optimization = \
        splitting_function(Datamanager.data['X_train'], Datamanager.data['Y_train'])
    X_valid = Datamanager.data.get('X_valid')
    X_test = Datamanager.data.get('X_test')

    if Datamanager.info['task'].lower() == 'regression':
        model = AutoSklearnRegressor(configuration, seed)
    else:
        model = AutoSklearnClassifier(configuration, seed)
    model.fit(X_train, Y_train)
    metric = Datamanager.info['metric']
    task_type = Datamanager.info['task']

    Y_optimization_pred = \
        predict_proba(X_optimization, model, task_type)
    if X_valid is not None:
        Y_valid_pred = predict_proba(X_valid, model, task_type)
    else:
        Y_valid_pred = None
    if X_test is not None:
        Y_test_pred = predict_proba(X_test, model, task_type)
    else:
        Y_test_pred = None

    score = calculate_score(Y_optimization, Y_optimization_pred,
                            task_type, metric,
                            all_scoring_functions=all_scoring_functions)
    if hasattr(score, "__len__"):
        err = {key: 1 - score[key] for key in score}
    else:
        err = 1 - score

    if with_predictions:
        return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
    else:
        return err
