'''
Created on Jan 7, 2015

@author: Aaron Klein
'''

import os
import sys
import glob
import numpy as np

from ensembles import weighted_ensemble, ensemble_prediction
from util.get_dataset_info import getInfoFromFile
from models import evaluate


def load_predictions(dirs):
    pred = []
    pred_valid = []
    pred_test = []
    for d in dirs:
        dir_test = os.path.join(d, "predictions_test/")
        dir_valid = os.path.join(d, "predictions_valid/")
        dir_ensemble = os.path.join(d, "predictions_ensemble/")
        for f in os.listdir(dir_ensemble):
            p = np.load(os.path.join(dir_ensemble, f))
            if not np.isfinite(p).all():
                continue
            pred.append(p)

            p = np.load(os.path.join(dir_valid, f.replace("ensemble", "valid")))
            pred_valid.append(p)

            p = np.load(os.path.join(dir_test, f.replace("ensemble", "test")))
            pred_test.append(p)

    assert len(pred) > 0

    return np.array(pred), np.array(pred_valid), np.array(pred_test)


def load_predictions_of_nbest(dirs, nbest, labels, task_type, metric):
    pred = []
    pred_valid = []
    pred_test = []
    for i in range(0, nbest):
        pred.append(0)
        pred_valid.append(0)
        pred_test.append(0)

    performance_nbest = np.ones([nbest]) * sys.float_info.max
    for d in dirs:
        dir_test = os.path.join(d, "predictions_test/")
        dir_valid = os.path.join(d, "predictions_valid/")
        dir_ensemble = os.path.join(d, "predictions_ensemble/")
        for f in os.listdir(dir_ensemble):
            p = np.load(os.path.join(dir_ensemble, f))
            if not np.isfinite(p).all():
                continue

            perf = 1 - evaluate.calculate_score(labels, p, task_type, metric)

            idx = np.argmax(performance_nbest)
            if(performance_nbest[idx] > perf):

                performance_nbest[idx] = perf
                pred[idx] = p

                p = np.load(os.path.join(dir_valid, f.replace("ensemble", "valid")))
                pred_valid[idx] = p
                p = np.load(os.path.join(dir_test, f.replace("ensemble", "test")))
                pred_test[idx] = p
            else:
                continue

    assert len(pred) > 0

    print performance_nbest
    return np.array(pred), np.array(pred_valid), np.array(pred_test)


def pick_nbest_models(pred, labels, task_type, metric, n_best=10):
    perf = np.zeros([pred.shape[0]])
    for i in range(0, pred.shape[0]):
        perf[i] = 1 - evaluate.calculate_score(labels, pred[i], task_type, metric)
    idx = np.argsort(perf)[:n_best]
    return idx


def pick_best_models(pred, labels, task_type, metric):
    best = []
    perf = np.zeros([pred.shape[0]])
    for i in range(0, pred.shape[0]):
        perf[i] = 1 - evaluate.calculate_score(labels, pred[i], task_type, metric)
        if perf[i] < 1.0:
            best.append(i)
    return np.array(best)


def main(dataset):
    print "Use data set: "
    print dataset
    print "Load predictions ..."
    path = "/home/feurerm/projects/automl_competition_2015/code/benchmarks/" + dataset + "/"

    dirs = glob.glob(path + "smac_2_08_00-*")
    #predictions, predictions_valid, predictions_test = load_predictions(dirs)

    print "Load labels ..."
    info = getInfoFromFile("/data/aad/automl_data/", dataset)
    true_labels = np.load(os.path.join(path, dataset + ".npy"))

    n_best = 10

    predictions, predictions_valid, predictions_test = load_predictions_of_nbest(dirs, n_best, true_labels, info['task'], info['metric'])

    print predictions.shape
    print predictions_valid.shape
    print predictions_test.shape
    print "Start optimization"
    weights = weighted_ensemble(predictions, true_labels, info['task'], info['metric'])
    print "finished"

    print "Weights: " + str(weights)

    ##=== Compute ensembles predictions for valid data
    Y_valid = ensemble_prediction(predictions_valid, weights)

    ##=== Compute ensembles predictions for test data
    Y_test = ensemble_prediction(predictions_test, weights)

    np.savetxt("predictions/" + dataset + "_valid_000.predict", Y_valid, delimiter=' ')
    np.savetxt("predictions/" + dataset + "_test_000.predict", Y_test, delimiter=' ')
    print "Predictions saved"

if __name__ == '__main__':
    #dataset = ["adult", "digits", "newsgroups", "dorothea", "cadata"]
    dataset = ["digits"]
    for d in dataset:
        main(d)
