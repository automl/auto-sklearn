'''
Created on Jan 7, 2015

@author: Aaron Klein
'''

import os
import glob
import numpy as np

from ensembles import weighted_ensemble, ensemble_prediction
from util.get_dataset_info import getInfoFromFile


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


def main(dataset):

    print "Load predictions ..."
    path = "/home/feurerm/projects/automl_competition_2015/code/benchmarks/" + dataset + "/"

    dirs = glob.glob(path + "smac_2_08_00-*")
    predictions, predictions_valid, predictions_test = load_predictions(dirs)

    print "Load labels ..."
    info = getInfoFromFile("/data/aad/automl_data/", dataset)
    true_labels = np.load(os.path.join(path, dataset + ".npy"))

    print "Start optimization"
    weights = weighted_ensemble(predictions, true_labels, info)
    print "finished"

    print "Compute predictions"
    ##=== Compute ensembles predictions for valid data
    Y_valid = ensemble_prediction(predictions_valid, weights)

    ##=== Compute ensembles predictions for test data
    Y_test = ensemble_prediction(predictions_test, weights)

    np.savetxt(dataset + "_valid_000.predict", Y_valid, delimiter=' ')
    np.savetxt(dataset + "_test_000.predict", Y_test, delimiter=' ')

if __name__ == '__main__':
    dataset = ["adult"]#, "dorothea", "digits", "newsgroups"]
    for d in dataset:
        main(d)
