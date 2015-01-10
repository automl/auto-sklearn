'''
Created on Jan 7, 2015

@author: Aaron Klein
'''

import os
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


def pick_nbest_models(pred, labels, task_type, metric, n_best=10):
    perf = np.zeros([pred.shape[0]])
    for i in range(0, pred.shape[0]):
        perf[i] = 1 - evaluate.calculate_score(labels, pred[i], task_type, metric)
        print perf[i]
    idx = np.argsort(perf)[:n_best]
    print "perf of best:"
    print perf[idx]
    return idx


def main(dataset):
    print dataset
    print "Load predictions ..."
    path = "/home/feurerm/projects/automl_competition_2015/code/benchmarks/" + dataset + "/"

    dirs = glob.glob(path + "smac_2_08_00-*")
    predictions, predictions_valid, predictions_test = load_predictions(dirs)

    print "Load labels ..."
    info = getInfoFromFile("/data/aad/automl_data/", dataset)
    true_labels = np.load(os.path.join(path, dataset + ".npy"))

    n_best = 10
    print "Pick best " + str(n_best) + " models"
    best = pick_nbest_models(predictions, true_labels, info['task'], info['metric'], n_best)

    predictions = predictions[best]
    predictions_valid = predictions_valid[best]
    predictions_test = predictions_test[best]

    print "Start optimization"
    weights = weighted_ensemble(predictions, true_labels, info['task'], info['metric'])
    print "finished"

    print "Compute predictions"
    ##=== Compute ensembles predictions for valid data
    Y_valid = ensemble_prediction(predictions_valid, weights)

    ##=== Compute ensembles predictions for test data
    Y_test = ensemble_prediction(predictions_test, weights)

    np.savetxt(dataset + "_valid_000.predict", Y_valid, delimiter=' ')
    np.savetxt(dataset + "_test_000.predict", Y_test, delimiter=' ')

if __name__ == '__main__':
    #dataset = ["adult", "digits", "newsgroups", "dorothea", "cadata"]
    dataset = ["cadata"]
    for d in dataset:
        main(d)
